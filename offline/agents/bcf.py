"""Behaviour Clone Flow (BCF) agent for OGBench.

Trains a flow-matching BC policy offline. The final checkpoint is saved under
exp/bcf/<env_name>/ and loaded by LARBRIDAgent as a frozen prior for noise steering.
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField


class BCFAgent(flax.struct.PyTreeNode):
    """Behaviour Clone Flow agent: trains a flow-matching BC policy offline.

    Fields:
        rng:      JAX PRNG key.
        bc_flow:  TrainState for the BC flow policy (actor_bc_flow).
        config:   Frozen hyper-parameter dict.
    """

    rng: Any
    bc_flow: Any
    config: Any = nonpytree_field()

    def _bc_flow_euler(self, observations, noises):
        """Euler integration of the BC flow policy for flow_steps steps."""
        actions = noises
        n = self.config['flow_steps']
        for i in range(n):
            t = jnp.full((*observations.shape[:-1], 1), i / n)
            vels = self.bc_flow.select('actor_bc_flow')(observations, actions, t)
            actions = actions + vels / n
        return jnp.clip(actions, -1.0, 1.0)

    def _bc_flow_loss(self, batch, grad_params, rng):
        """Flow-matching loss (FQL-style)."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.bc_flow.select('actor_bc_flow')(
            batch['observations'], x_t, t, params=grad_params
        )
        loss = jnp.mean((pred - vel) ** 2)
        return loss, {'bc_flow/loss': loss}

    @jax.jit
    def update(self, batch):
        """Offline update: trains the BC flow policy."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self._bc_flow_loss(batch, grad_params, rng)

        new_bc_flow, info = self.bc_flow.apply_loss_fn(loss_fn)
        return self.replace(rng=new_rng, bc_flow=new_bc_flow), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Sample actions via Euler integration with Gaussian prior noise."""
        _, noise_rng = jax.random.split(seed)
        noise_shape = observations.shape[:-len(self.config['ob_dims'])] + (self.config['action_dim'],)
        noises = jax.random.normal(noise_rng, noise_shape)
        return self._bc_flow_euler(observations, noises)

    @jax.jit
    def total_loss(self, batch, grad_params=None, rng=None):
        """Validation loss (does not update parameters)."""
        rng = rng if rng is not None else self.rng
        _, bc_rng = jax.random.split(rng)
        _, info = self._bc_flow_loss(batch, self.bc_flow.params, bc_rng)
        return 0.0, info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, bc_rng = jax.random.split(rng)

        action_dim = ex_actions.shape[-1]
        ex_times = ex_actions[..., :1]

        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor_bc_flow'] = encoder_module()

        bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        bc_flow_module = ModuleDict({'actor_bc_flow': bc_flow_def})
        bc_flow_params = bc_flow_module.init(
            bc_rng,
            actor_bc_flow=(ex_observations, ex_actions, ex_times),
        )['params']
        bc_flow = TrainState.create(
            bc_flow_module,
            bc_flow_params,
            tx=optax.adam(learning_rate=config['lr']),
        )

        config_dict = {
            'agent_name': 'bcf',
            'lr': float(config['lr']),
            'batch_size': int(config['batch_size']),
            'actor_hidden_dims': tuple(config['actor_hidden_dims']),
            'actor_layer_norm': bool(config['actor_layer_norm']),
            'actor_fc_scale': float(config['actor_fc_scale']),
            'flow_steps': int(config['flow_steps']),
            'encoder': config['encoder'],
            'ob_dims': tuple(ex_observations.shape[1:]),
            'action_dim': int(action_dim),
        }

        return cls(
            rng=rng,
            bc_flow=bc_flow,
            config=flax.core.FrozenDict(**config_dict),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='bcf',
            lr=3e-4,
            batch_size=256,
            encoder=ml_collections.config_dict.placeholder(str),
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            actor_fc_scale=0.01,
            flow_steps=10,
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
