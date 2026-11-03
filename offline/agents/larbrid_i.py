"""LARBRID-I agent.

Extends FQL with a DiffSR component that learns low-rank MDP feature
representations via diffusion score matching. The critic operates on
phi(s, a) features rather than raw (s, a) inputs, following the linear
MDP parameterisation from LARBRID.

Architecture (single TrainState, flat ModuleDict — mirrors fql.py):
  - diffsr_phi:     PhiNet mapping (s, a) -> (B, F)
  - diffsr_mu:      MuNet mapping (s_noisy, t) -> (B, F, obs_dim)
  - diffsr_reward:  RffRewardNet mapping phi -> scalar reward prediction
  - critic:         Value network mapping phi -> (2, B) Q values
  - actor_bc_flow:  ActorVectorField for BC flow matching
  - actor_onestep_flow: ActorVectorField for one-step distillation
  - target_* copies of diffsr_phi/mu/reward and critic (Polyak-averaged)

Loss components:
  - diffsr_loss: score matching (phi @ mu ≈ eps) + reward prediction
  - critic_loss: TD backup using target_phi and target_critic
  - actor_loss:  BC flow + distillation + Q via phi (same as FQL)
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.modules.diffsr import MuNet, PhiNet
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, RffRewardNet, Value


def _vp_alphabars(sample_steps: int):
    """Variance-preserving cumulative product ᾱ_t for t = 1…T.

    alpha_t = exp(-beta_min/T - 0.5*(beta_max-beta_min)*(2t-1)/T^2)
    ᾱ_t     = prod_{i=1}^{t} alpha_i   (decreases from ~1 toward ~0)

    NOTE: the variable named 'alpha' here equals (1 - beta_t) from the VP-SDE
    schedule, so alphabar = cumprod(alpha), NOT cumprod(1-alpha).
    cumprod(1-alpha) = cumprod(betas) collapses to ≈ 0 immediately and
    makes every noisy sample pure noise, killing the score-matching signal.
    """
    t = jnp.arange(1, sample_steps + 1, dtype=jnp.float32)
    b_max, b_min = 10.0, 0.1
    alpha = jnp.exp(-b_min / sample_steps - 0.5 * (b_max - b_min) * (2 * t - 1) / sample_steps ** 2)
    return jnp.cumprod(alpha)  # (T,)


class LARBRIDIAgent(flax.struct.PyTreeNode):
    """LARBRID-I agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def diffsr_loss(self, batch, grad_params, rng):
        """DiffSR score matching + reward prediction loss."""
        s = batch['observations']
        a = batch['actions']
        sp = batch['next_observations']
        r = batch['rewards']

        rng, t_rng, eps_rng = jax.random.split(rng, 3)

        T = self.config['diffsr_sample_steps']
        t_idx = jax.random.randint(t_rng, (s.shape[0],), 0, T)

        alphabars = _vp_alphabars(T)
        ab = alphabars[t_idx]  # (B,)
        eps = jax.random.normal(eps_rng, sp.shape)  # (B, obs_dim)
        sp_noisy = jnp.sqrt(ab[:, None]) * sp + jnp.sqrt(1.0 - ab[:, None]) * eps

        # phi: (B, F),  mu: (B, F, obs_dim) — gradients flow to diffsr params
        phi = self.network.select('diffsr_phi')(s, a, params=grad_params)
        mu = self.network.select('diffsr_mu')(
            sp_noisy, t_idx[:, None].astype(jnp.float32), params=grad_params
        )

        eps_pred = jnp.einsum('bf,bfo->bo', phi, mu)
        # sum over obs_dim first, then mean over batch — matches reference .pow(2).sum(-1).mean()
        diffusion_loss = jnp.mean(jnp.sum((eps_pred - eps) ** 2, axis=-1))

        reward_pred = self.network.select('diffsr_reward')(phi, params=grad_params)
        reward_loss = jnp.mean((reward_pred - r) ** 2)

        total = diffusion_loss + reward_loss
        return total, {
            'total_loss': total,
            'diffusion_loss': diffusion_loss,
            'reward_loss': reward_loss,
            'phi_norm': jnp.mean(jnp.linalg.norm(phi, axis=-1)),
        }

    def critic_loss(self, batch, grad_params, rng):
        """TD critic loss using phi features.

        TD target:  target_phi(s', a') -> target_critic -> Q_target
        Current Q:  phi(s, a) [stop-grad] -> critic [grad] -> Q
        """
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        # Target phi and Q — no grad_params, so phi/critic weights are fixed here
        next_phi = self.network.select('target_diffsr_phi')(batch['next_observations'], next_actions)
        next_qs = self.network.select('target_critic')(next_phi)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        # Current phi from target (EMA) network — matches reference which uses ddpm_target
        # for both the TD-target and the current Q to keep the feature space stable.
        current_phi = self.network.select('target_diffsr_phi')(batch['observations'], batch['actions'])
        # Critic gets gradient via grad_params
        q = self.network.select('critic')(current_phi, params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """FQL actor loss: BC flow + distillation + Q via phi."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss via target phi — stable EMA representation, matching reference.
        # Gradients flow through target_phi's computation to actor_actions (chain rule),
        # but target_phi weights are not part of grad_params.
        actor_actions_clipped = jnp.clip(actor_actions, -1, 1)
        actor_phi = self.network.select('target_diffsr_phi')(batch['observations'], actor_actions_clipped)
        qs = self.network.select('critic')(actor_phi)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute total loss: diffsr + critic + actor."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, diffsr_rng, critic_rng, actor_rng = jax.random.split(rng, 4)

        diffsr_loss, diffsr_info = self.diffsr_loss(batch, grad_params, diffsr_rng)
        for k, v in diffsr_info.items():
            info[f'diffsr/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = diffsr_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Polyak (EMA) update for a target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with info dict."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        self.target_update(new_network, 'diffsr_phi')
        self.target_update(new_network, 'diffsr_mu')
        self.target_update(new_network, 'diffsr_reward')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Sample actions from the one-step policy."""
        action_seed, _ = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new LARBRIDIAgent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        obs_dim = ex_observations.shape[-1]

        # Encoders (actor only — phi operates on state-space observations)
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # DiffSR networks
        phi_def = PhiNet(
            embed_dim=config['diffsr_embed_dim'],
            feature_dim=config['diffsr_feature_dim'],
            hidden_dims=config['diffsr_phi_hidden_dims'],
        )
        mu_def = MuNet(
            obs_dim=obs_dim,
            embed_dim=config['diffsr_embed_dim'],
            feature_dim=config['diffsr_feature_dim'],
            hidden_dims=config['diffsr_mu_hidden_dims'],
        )
        reward_def = RffRewardNet(
            rff_dim=config['diffsr_rff_dim'],
            hidden_dims=config['diffsr_reward_hidden_dims'],
        )

        ex_t_diffsr = jnp.zeros((ex_observations.shape[0], 1))
        ex_phi = jnp.zeros((ex_observations.shape[0], config['diffsr_feature_dim']))

        # Critic takes phi features as input (no actions arg)
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )

        # Actor networks (same as FQL)
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            diffsr_phi=(phi_def, (ex_observations, ex_actions)),
            target_diffsr_phi=(copy.deepcopy(phi_def), (ex_observations, ex_actions)),
            diffsr_mu=(mu_def, (ex_observations, ex_t_diffsr)),
            target_diffsr_mu=(copy.deepcopy(mu_def), (ex_observations, ex_t_diffsr)),
            diffsr_reward=(reward_def, (ex_phi,)),
            target_diffsr_reward=(copy.deepcopy(reward_def), (ex_phi,)),
            critic=(critic_def, (ex_phi,)),
            target_critic=(copy.deepcopy(critic_def), (ex_phi,)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders['actor_bc_flow'], (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        # DiffSR params use a separate (lower) learning rate
        _diffsr_param_keys = {
            'modules_diffsr_phi',
            'modules_diffsr_mu',
            'modules_diffsr_reward',
        }

        def param_labels(params):
            return {
                k: jax.tree_util.tree_map(
                    lambda _: 'diffsr' if k in _diffsr_param_keys else 'main',
                    v,
                )
                for k, v in params.items()
            }

        network_tx = optax.multi_transform(
            {
                'diffsr': optax.adam(learning_rate=config['diffsr_lr']),
                'main': optax.adam(learning_rate=config['lr']),
            },
            param_labels=param_labels,
        )

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Initialise target networks as copies of their live counterparts
        params = network.params
        params['modules_target_diffsr_phi'] = params['modules_diffsr_phi']
        params['modules_target_diffsr_mu'] = params['modules_diffsr_mu']
        params['modules_target_diffsr_reward'] = params['modules_diffsr_reward']
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='larbrid_i',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),

            # Shared
            lr=3e-4,
            batch_size=256,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            encoder=ml_collections.config_dict.placeholder(str),

            # Actor (same as FQL)
            actor_hidden_dims=(512, 512, 512, ), # 512, 512, 512, 512
            actor_layer_norm=False,
            alpha=10.0,
            flow_steps=10,
            normalize_q_loss=False,

            # Critic (operates on phi features)
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,

            # DiffSR
            diffsr_lr=3e-4,
            diffsr_embed_dim=256,
            diffsr_feature_dim=512,
            diffsr_phi_hidden_dims=(512,),
            diffsr_mu_hidden_dims=(512,),
            diffsr_reward_hidden_dims=(512, ),
            diffsr_rff_dim=1024,
            diffsr_sample_steps=50,
        )
    )
    return config
