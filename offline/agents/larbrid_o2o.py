"""LARBRID agent for OGBench.

Supports two frozen prior policy types (configured via prior_policy_type):
  - 'bcf': multi-step Euler integration through BCFAgent's actor_bc_flow network.
           Pass bcf_agent.bc_flow to LARBRIDAgent.create(bc_flow=...).
  - 'fql': single forward pass through FQLAgent's distilled actor_onestep_flow network.
           Pass fql_agent.network to LARBRIDAgent.create(bc_flow=...).

  - DiffSR (phi/mu/reward) learns a single set of low-rank MDP features via
    score matching.  A single phi representation is shared across all critics.
  - Critic Q(phi) is trained via TD backup: N critic ensemble heads all receive
    the same phi(s, a) feature, providing ensemble diversity purely in the critic.
  - NoiseFeature bridge maps (obs, noise) → phi to avoid diffusion backprop.
    N independent noise-feat heads, each supervised by the single phi target.
  - Noise actor π^w optimizes latent noise to maximise Q via the bridge.

Key design choices:
  - Four trainable TrainState objects (diffsr, critic, noise_feat, noise_actor)
    with independent Adam optimisers.
  - bc_flow is stored as a frozen TrainState (tx=None); never updated.
  - Target networks (target_diffsr, target_critic, target_noise_actor) stored as
    TrainState with tx=None; Polyak-averaged after each gradient step.
  - Noise actor uses a SAC-style Gaussian policy (Actor) with auto entropy (LogParam α).
  - All stop-gradient semantics are enforced by calling networks without params=grad_params.
  - DiffSR phi/mu/reward and NoiseFeature are single networks; output shapes (B, F).
  - Critic split_input=False; all N heads receive the same (B, F) phi/feat via in_axes=None.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.modules.diffsr import MuNet, PhiNet
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, LogParam, NoiseFeature, RffRewardNet, Value


# ---------------------------------------------------------------------------
# VP noise schedule (pure JAX — safe inside @jax.jit)
# ---------------------------------------------------------------------------

def _vp_alphabars(sample_steps: int):
    """Variance-preserving cumulative product ᾱ_t for t = 1…T."""
    t = jnp.arange(1, sample_steps + 1, dtype=jnp.float32)
    b_max, b_min = 10.0, 0.1
    alpha = jnp.exp(-b_min / sample_steps - 0.5 * (b_max - b_min) * (2 * t - 1) / sample_steps ** 2)
    return jnp.cumprod(alpha)  # (T,)


# ---------------------------------------------------------------------------
# LARBRIDAgent
# ---------------------------------------------------------------------------

class LARBRIDO2OAgent(flax.struct.PyTreeNode):
    """LARBRID agent using a frozen BC flow policy loaded from a BCFAgent checkpoint.

    Separate TrainState objects are kept for each learnable component so that
    gradient updates can be applied independently.

    Fields:
        rng:               JAX PRNG key.
        bc_flow:           Frozen TrainState for the BC flow policy (tx=None, never updated).
        diffsr:            TrainState for DiffSR (single phi, mu, reward — no ensemble).
        target_diffsr:     Polyak-averaged copy of diffsr (no optimizer).
        critic:            TrainState for the feature critic Q(phi), split_input=False.
        target_critic:     Polyak-averaged copy of critic (no optimizer).
        noise_feat:        TrainState for the NoiseFeature bridge (single network).
        target_noise_feat: Polyak-averaged copy of noise_feat (no optimizer).
        noise_actor:       TrainState for the noise actor π^w + entropy coeff α.
        target_noise_actor: Polyak-averaged copy of noise_actor (no optimizer).
        config:            Frozen hyper-parameter dict.
    """

    rng: Any
    bc_flow: Any
    diffsr: Any
    target_diffsr: Any
    critic: Any
    target_critic: Any
    noise_feat: Any
    target_noise_feat: Any
    noise_actor: Any
    target_noise_actor: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _polyak(self, current_params, target_params):
        """Polyak (EMA) update: τ * current + (1 - τ) * target."""
        tau = self.config['tau']
        return jax.tree_util.tree_map(
            lambda c, t: tau * c + (1.0 - tau) * t,
            current_params,
            target_params,
        )

    def _bc_flow_euler(self, observations, noises):
        """Run the frozen prior policy to produce actions from noises.

        prior_policy_type='bcf': multi-step Euler integration through actor_bc_flow
            (flow_steps Euler steps; BCFAgent checkpoint).
        prior_policy_type='fql': single forward pass through actor_onestep_flow
            (FQLAgent distilled checkpoint; flow_steps is ignored).

        No gradients flow to bc_flow (params called without grad_params).
        """
        if self.config['prior_policy_type'] == 'fql':
            actions = self.bc_flow.select('actor_onestep_flow')(observations, noises)
            return jnp.clip(actions, -1.0, 1.0)

        # BCF: multi-step Euler integration
        actions = noises
        n = self.config['flow_steps']
        for i in range(n):
            t = jnp.full((*observations.shape[:-1], 1), i / n)
            vels = self.bc_flow.select('actor_bc_flow')(observations, actions, t)
            actions = actions + vels / n
        return jnp.clip(actions, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Offline: DiffSR, critic, noise-feature, noise-actor losses
    # ------------------------------------------------------------------

    def _diffsr_loss(self, batch, grad_params, rng):
        """DiffSR loss: diffusion score matching + reward prediction.

        Single phi/mu/reward — no ensemble.
        phi: (B, F); mu: (B, F, obs_dim).
        Score matching: phi @ mu ≈ eps.
        """
        s = batch['observations']
        a = batch['actions']
        sp = batch['next_observations']
        r = batch['rewards']

        rng, t_rng, eps_rng = jax.random.split(rng, 3)

        T = self.config['diffsr_sample_steps']
        t_idx = jax.random.randint(t_rng, (s.shape[0],), 0, T)

        alphabars = _vp_alphabars(T)          # (T,)
        ab = alphabars[t_idx]                 # (B,)
        eps = jax.random.normal(eps_rng, sp.shape)
        sp_noisy = jnp.sqrt(ab[:, None]) * sp + jnp.sqrt(1.0 - ab[:, None]) * eps

        # phi: (B, F), mu: (B, F, obs_dim)
        phi = self.diffsr.select('phi')(s, a, params=grad_params)
        mu = self.diffsr.select('mu')(
            sp_noisy, t_idx[:, None].astype(jnp.float32), params=grad_params
        )

        # eps_pred = phi @ mu; einsum over feature dim F → (B, obs_dim)
        eps_pred = jnp.einsum('bf,bfo->bo', phi, mu)

        # standard noise matching diffusion loss
        diffusion_loss = jnp.mean(jnp.sum((eps_pred - eps) ** 2, axis=-1))

        # reward(phi): (B,)
        reward_pred = self.diffsr.select('reward')(phi, params=grad_params)
        reward_loss = jnp.mean((reward_pred - r) ** 2)

        total = diffusion_loss + reward_loss
        return total, {
            'diffsr/total_loss': total,
            'diffsr/diffusion_loss': diffusion_loss,
            'diffsr/reward_loss': reward_loss,
            'diffsr/phi_norm': jnp.mean(jnp.linalg.norm(phi, axis=-1)),
        }

    def _critic_loss(self, batch, grad_params, rng):
        """TD critic loss: all N critics receive the same (B, F) phi.

        next_phi/current_phi: (B, F); critic with split_input=False vmaps all N heads
        over the same input → next_qs/current_qs: (N, B).
        """
        s = batch['observations']
        a = batch['actions']
        sp = batch['next_observations']
        r = batch['rewards']
        masks = batch['masks']

        rng, next_rng = jax.random.split(rng)

        # Next noise from target noise actor (stop-grad)
        next_dist = self.target_noise_actor.select('actor')(sp)
        next_noises, next_log_probs = next_dist.sample_and_log_prob(seed=next_rng)

        # Frozen bc_flow maps next noise → next action (stop-grad on bc_flow)
        next_actions = self._bc_flow_euler(sp, next_noises)

        # Target features and Q (all stop-grad); next_phi: (B, F) → next_qs: (N, B)
        next_phi = self.target_diffsr.select('phi')(sp, next_actions)
        next_qs = self.target_critic.select('critic')(next_phi)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        alpha = self.noise_actor.select('alpha')()
        target_q = r + self.config['discount'] * masks * (next_q - alpha * next_log_probs)

        # Current phi from live diffsr (stop-grad on diffsr params); current_phi: (B, F)
        current_phi = self.target_diffsr.select('phi')(s, a)

        # Critic forward with gradient; current_qs: (N, B)
        current_qs = self.critic.select('critic')(current_phi, params=grad_params)
        # target_q: (B,) broadcast against (N, B)
        critic_loss = jnp.mean(jnp.square(current_qs - target_q))

        return critic_loss, {
            'critic/loss': critic_loss,
            'critic/q_mean': current_qs.mean(),
            'critic/q_max': current_qs.max(),
            'critic/q_min': current_qs.min(),
            'critic/target_q_mean': target_q.mean(),
            'critic/target_q_max': target_q.max(),
            'critic/target_q_min': target_q.min(),
        }

    def _noise_feat_loss(self, batch, grad_params, rng):
        """NoiseFeature distillation: noise_feat(s,w) ≈ phi(s, bc_flow(s,w)).

        Both pred_phi and target_phi are (B, F) — single networks, no ensemble.
        """
        s = batch['observations']
        rng, sample_rng = jax.random.split(rng)

        # Noises from target noise actor (stop-grad)
        w_dist = self.target_noise_actor.select('actor')(s)
        w, _ = w_dist.sample_and_log_prob(seed=sample_rng)

        # Frozen bc_flow → action (stop-grad)
        actions = self._bc_flow_euler(s, w)

        #target phi from single diffsr (stop-grad): (B, F)
        target_phi = self.diffsr.select('phi')(s, actions) # IMPORTANT: needs to compute from current feature, not target feature

        # Noise feature prediction with gradient: (B, F)
        pred_phi = self.noise_feat.select('noise_feat')(s, w, params=grad_params)

        loss = jnp.mean(jnp.square(pred_phi - target_phi))
        return loss, {
            'noise_feat/loss': loss,
            'noise_feat/pred_phi_norm': jnp.mean(jnp.linalg.norm(pred_phi, axis=-1)),
            'noise_feat/target_phi_norm': jnp.mean(jnp.linalg.norm(target_phi, axis=-1)),
        }

    def _actor_alpha_loss(self, batch, grad_params, rng):
        """SAC-style actor loss maximising Q^W via the noise-feature bridge.

        Gradient flows: noise_actor_params → w → noise_feat(obs, w) → critic → Q.
        feat: (B, F); critic with split_input=False vmaps all N heads → qs: (N, B).
        """
        s = batch['observations']
        rng, sample_rng = jax.random.split(rng)

        # Sample noise from current noise actor (gradient flows here)
        dist = self.noise_actor.select('actor')(s, params=grad_params)
        w, log_probs = dist.sample_and_log_prob(seed=sample_rng)

        # Single target_noise_feat (stop-grad on target params): (B, F)
        feat = self.target_noise_feat.select('noise_feat')(s, w)
        # critic with split_input=False; (N, B)
        qs = self.critic.select('critic')(feat)
        if self.config['q_agg'] == 'min':
            q = qs.min(axis=0)
        else:
            q = qs.mean(axis=0)

        # Entropy coeff with gradient
        alpha = self.noise_actor.select('alpha')(params=grad_params)
        actor_loss = (jax.lax.stop_gradient(alpha) * log_probs - q).mean()

        # Alpha loss: push entropy toward target.
        # log_probs must be stop-gradiented so alpha_loss does not send gradients
        # back through the actor params — mirrors the reference where log_prob is
        # explicitly detached before computing the entropy-coefficient loss.
        target_entropy = self.config['target_entropy']
        alpha_loss = (alpha * jax.lax.stop_gradient(-log_probs - target_entropy).mean()).mean()

        total = actor_loss + alpha_loss
        return total, {
            'actor/loss': actor_loss,
            'actor/alpha_loss': alpha_loss,
            'actor/alpha': alpha,
            'actor/entropy': -log_probs.mean(),
            'actor/q_mean': q.mean(),
            'actor/q_std': q.std(),
        }

    @jax.jit
    def update(self, batch):
        """Online update: DiffSR → Critic → NoiseFeature (×K) → Actor (separate grad steps).

        NoiseFeature is updated noise_feat_update_steps times per call, each with a
        fresh RNG key so the actor samples different noises, giving better coverage of
        the noise-to-phi mapping without changing the rest of the pipeline.
        bc_flow is frozen throughout; target networks are Polyak-updated each step.
        """
        new_rng, rng = jax.random.split(self.rng)
        rng, diffsr_rng, critic_rng, nf_rng, actor_rng = jax.random.split(rng, 5)

        # 1. Update DiffSR
        new_diffsr, diffsr_info = self.diffsr.apply_loss_fn(
            lambda p: self._diffsr_loss(batch, p, diffsr_rng)
        )

        # 2. Update Critic
        new_critic, critic_info = self.critic.apply_loss_fn(
            lambda p: self._critic_loss(batch, p, critic_rng)
        )

        # 3. Update NoiseFeature × noise_feat_update_steps
        # Each step draws fresh noise from the actor via a distinct RNG key,
        # giving different (obs, noise) pairs even though the obs batch is fixed.
        nf_steps = self.config['noise_feat_update_steps']
        nf_rngs = jax.random.split(nf_rng, nf_steps)

        def _nf_step(noise_feat, step_rng):
            new_nf, info = noise_feat.apply_loss_fn(
                lambda p: self._noise_feat_loss(batch, p, step_rng)
            )
            return new_nf, info

        new_noise_feat, nf_infos = jax.lax.scan(_nf_step, self.noise_feat, nf_rngs)
        # Report metrics from the final update step
        nf_info = jax.tree_util.tree_map(lambda x: x[-1], nf_infos)

        # 4. Update NoiseActor + alpha
        new_noise_actor, actor_info = self.noise_actor.apply_loss_fn(
            lambda p: self._actor_alpha_loss(batch, p, actor_rng)
        )

        # 5. Polyak update of target networks
        new_target_diffsr = self.target_diffsr.replace(
            params=self._polyak(new_diffsr.params, self.target_diffsr.params)
        )
        new_target_critic = self.target_critic.replace(
            params=self._polyak(new_critic.params, self.target_critic.params)
        )
        new_target_noise_feat = self.target_noise_feat.replace(
            params=self._polyak(new_noise_feat.params, self.target_noise_feat.params)
        )
        new_target_noise_actor = self.target_noise_actor.replace(
            params=self._polyak(new_noise_actor.params, self.target_noise_actor.params)
        )

        info = {}
        for d in (diffsr_info, critic_info, nf_info, actor_info):
            info.update(d)

        return self.replace(
            rng=new_rng,
            diffsr=new_diffsr,
            target_diffsr=new_target_diffsr,
            critic=new_critic,
            target_critic=new_target_critic,
            noise_feat=new_noise_feat,
            target_noise_feat=new_target_noise_feat,
            noise_actor=new_noise_actor,
            target_noise_actor=new_target_noise_actor,
        ), info

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Sample actions using noise_actor → frozen bc_flow Euler integration."""
        rng, sample_rng = jax.random.split(seed)
        dist = self.noise_actor.select('actor')(observations, temperature=temperature)
        noises = dist.sample(seed=sample_rng)
        return self._bc_flow_euler(observations, noises)

    # ------------------------------------------------------------------
    # Validation loss (used in main.py logging)
    # ------------------------------------------------------------------

    @jax.jit
    def total_loss(self, batch, grad_params=None, rng=None):
        """Return validation losses for logging (does not update parameters)."""
        rng = rng if rng is not None else self.rng
        _, diffsr_rng = jax.random.split(rng)
        _, diffsr_info = self._diffsr_loss(batch, self.diffsr.params, diffsr_rng)
        return 0.0, diffsr_info

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, bc_flow=None):
        """Initialise all networks and return a fresh LARBRIDO2OAgent.

        Args:
            seed:            Integer random seed.
            ex_observations: Example observation batch (shape: [1, obs_dim]).
            ex_actions:      Example action batch (shape: [1, action_dim]).
            config:          ml_collections.ConfigDict with hyper-parameters.
                             Must include prior_policy_type ('bcf' or 'fql').
            bc_flow:         Frozen prior policy TrainState. What to pass depends on
                             prior_policy_type:
                               'bcf' → bcf_agent.bc_flow  (BCFAgent checkpoint;
                                       ModuleDict with actor_bc_flow)
                               'fql' → fql_agent.network  (FQLAgent checkpoint;
                                       ModuleDict with actor_onestep_flow)
        """
        if bc_flow is None:
            raise ValueError(
                "LARBRIDO2OAgent requires a pre-trained prior policy. "
                "For prior_policy_type='bcf' pass bcf_agent.bc_flow; "
                "for prior_policy_type='fql' pass fql_agent.network."
            )

        rng = jax.random.PRNGKey(seed)
        rng, online_rng = jax.random.split(rng)

        obs_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]
        N = config['num_ensembles']

        # ------------------------------------------------------------------
        # Optionally set target_entropy from action_dim
        # ------------------------------------------------------------------
        if config['target_entropy'] is None:
            target_entropy = -config['target_entropy_multiplier'] * action_dim
        else:
            target_entropy = float(config['target_entropy'])
        config = config.copy_and_resolve_references()

        # ------------------------------------------------------------------
        # Freeze bc_flow: strip optimizer so it is never updated.
        # ------------------------------------------------------------------
        bc_flow_frozen = TrainState.create(bc_flow.model_def, bc_flow.params, tx=None)

        # ------------------------------------------------------------------
        # 1. DiffSR — single phi, mu, reward (no ensemble)
        #
        # phi: (B, F); mu: (B, F, obs_dim); reward: (B,).
        # ------------------------------------------------------------------
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

        ex_sp_noisy = ex_observations
        ex_t = jnp.zeros((ex_observations.shape[0], 1))
        # Single phi example: (B, F) — used to init reward and critic
        ex_phi = jnp.zeros((ex_observations.shape[0], config['diffsr_feature_dim']))

        diffsr_module = ModuleDict({
            'phi': phi_def,
            'mu': mu_def,
            'reward': reward_def,
        })
        diffsr_params = diffsr_module.init(
            online_rng,
            phi=(ex_observations, ex_actions),
            mu=(ex_sp_noisy, ex_t),
            reward=(ex_phi,),
        )['params']

        diffsr = TrainState.create(
            diffsr_module,
            diffsr_params,
            tx=optax.chain(
                optax.clip_by_global_norm(config['max_grad_norm']),
                optax.adam(learning_rate=config['diffsr_lr']),
            ),
        )
        target_diffsr = TrainState.create(diffsr_module, copy.deepcopy(diffsr_params), tx=None)

        # ------------------------------------------------------------------
        # 2. Feature Critic Q(phi) — N ensemble heads, split_input=False
        #
        # All N heads receive the same (B, F) phi via in_axes=None vmap.
        # ------------------------------------------------------------------
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=N,
            encoder=None,
            split_input=False,
        )
        critic_module = ModuleDict({'critic': critic_def})
        critic_params = critic_module.init(
            online_rng,
            critic=(ex_phi,),
        )['params']
        critic = TrainState.create(
            critic_module,
            critic_params,
            tx=optax.chain(
                optax.clip_by_global_norm(config['max_grad_norm']),
                optax.adam(learning_rate=config['lr']),
            ),
        )
        target_critic = TrainState.create(critic_module, copy.deepcopy(critic_params), tx=None)

        # ------------------------------------------------------------------
        # 3. NoiseFeature — single network (no ensemble)
        #
        # Output: (B, F); passed directly to the split_input=False critic.
        # ------------------------------------------------------------------
        noise_feat_def = NoiseFeature(
            embed_dim=config['diffsr_embed_dim'],
            feature_dim=config['diffsr_feature_dim'],
            phi_hidden_dims=config['diffsr_phi_hidden_dims'],
        )
        noise_feat_module = ModuleDict({'noise_feat': noise_feat_def})
        noise_feat_params = noise_feat_module.init(
            online_rng,
            noise_feat=(ex_observations, ex_actions),
        )['params']
        noise_feat = TrainState.create(
            noise_feat_module,
            noise_feat_params,
            tx=optax.chain(
                optax.clip_by_global_norm(config['max_grad_norm']),
                optax.adam(learning_rate=config['lr']),
            ),
        )
        target_noise_feat = TrainState.create(
            noise_feat_module, copy.deepcopy(noise_feat_params), tx=None
        )

        # ------------------------------------------------------------------
        # 4. Noise actor π^w + entropy coeff α
        # ------------------------------------------------------------------
        noise_actor_def = Actor(
            hidden_dims=config['noise_actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
        )
        alpha_def = LogParam(init_value=1.0)

        noise_actor_module = ModuleDict({'actor': noise_actor_def, 'alpha': alpha_def})
        noise_actor_params = noise_actor_module.init(
            online_rng,
            actor=(ex_observations,),
            alpha=(),
        )['params']
        noise_actor = TrainState.create(
            noise_actor_module,
            noise_actor_params,
            tx=optax.chain(
                optax.clip_by_global_norm(config['max_grad_norm']),
                optax.adam(learning_rate=config['lr']),
            ),
        )
        target_noise_actor = TrainState.create(
            noise_actor_module,
            copy.deepcopy(noise_actor_params),
            tx=None,
        )

        # Write resolved target_entropy into a mutable config copy
        config_dict = dict(config)
        config_dict['target_entropy'] = target_entropy
        config_dict['ob_dims'] = ex_observations.shape[1:]
        config_dict['action_dim'] = action_dim

        return cls(
            rng=rng,
            bc_flow=bc_flow_frozen,
            diffsr=diffsr,
            target_diffsr=target_diffsr,
            critic=critic,
            target_critic=target_critic,
            noise_feat=noise_feat,
            target_noise_feat=target_noise_feat,
            noise_actor=noise_actor,
            target_noise_actor=target_noise_actor,
            config=flax.core.FrozenDict(**config_dict),
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='larbrid_o2o',

            # -- Shared --
            lr=3e-4,                         # Adam learning rate for all non-DiffSR networks.
            max_grad_norm=10.0,
            batch_size=256,
            discount=0.99,
            tau=0.005,                        # Polyak averaging rate for all target networks.
            encoder=ml_collections.config_dict.placeholder(str),

            # -- Prior policy (frozen; loaded from BCFAgent or FQLAgent checkpoint) --
            # 'bcf': multi-step Euler through actor_bc_flow (pass bcf_agent.bc_flow)
            # 'fql': single forward pass through a distilled actor_onestep_flow (pass fql_agent.network)
            prior_policy_type='bcf',
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            actor_fc_scale=1e-4,              # Zero-init final layers → N(0,I) at start.
            flow_steps=10,                   # Euler steps for bc_flow inference (bcf only).

            # -- Ensembles (critic only; diffsr and noise_feat are single networks) --
            num_ensembles=2,
            value_hidden_dims=(512, 512, 512,),
            layer_norm=True,
            q_agg = "min",

            # -- DiffSR --
            diffsr_lr=1e-4,                  # Separate LR for DiffSR (from LARBRID codebase).
            diffsr_embed_dim=256,            # Embedding dim for obs/action/time.
            diffsr_feature_dim=512,          # Phi feature dimension.
            diffsr_phi_hidden_dims = (512,),
            diffsr_mu_hidden_dims = (512,),
            diffsr_reward_hidden_dims = (512, ),
            diffsr_rff_dim=1024,              # RFF dim for reward head.
            diffsr_sample_steps=50,         # DDPM noise schedule length.

            # -- Noise actor --
            noise_actor_hidden_dims=(512, 512, 512),
            tanh_squash=True,

            # -- NoiseFeature training --
            noise_feat_update_steps=1,           # Gradient steps on noise_feat per update call.

            # -- Noise actor entropy --
            target_entropy=0,
            target_entropy_multiplier=1,   # target_entropy = -multiplier * action_dim.

            # Resolved at creation time; included here so config is self-contained.
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
