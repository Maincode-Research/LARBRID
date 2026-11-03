from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp


def mish(x):
    """Mish activation: x * tanh(softplus(x))."""
    return x * jnp.tanh(jax.nn.softplus(x))


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        # Use zeros initializer when scale==0 so output is exactly 0 at init,
        # giving mean=0 and log_std=0 (std=1), i.e. N(0,I) at initialisation.
        fc_init = (
            nn.initializers.zeros
            if self.final_fc_init_scale == 0.0
            else default_init(self.final_fc_init_scale)
        )
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=fc_init)
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=fc_init)
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
        split_input: If True, each ensemble member receives its own input slice (in_axes=0).
                     Use when inputs are pre-stacked as (N, B, F) with one slice per head.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None
    split_input: bool = False

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            in_axes = 0 if self.split_input else None
            mlp_class = ensemblize(mlp_class, self.num_ensembles, in_axes=in_axes)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net
        #self.input_norm = nn.LayerNorm()

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        #inputs = self.input_norm(inputs)

        v = self.value_net(inputs).squeeze(-1)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v


class ResidualMLPBlock(nn.Module):
    """Residual MLP block with LayerNorm and Mish activation.

    Matches the ResidualMLPBlock from the LARBRID PyTorch implementation.

    Attributes:
        hidden_dim: Hidden/output dimension of this block.
    """

    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = mish(x)
        x = nn.Dense(self.hidden_dim)(x)
        if input_dim != self.hidden_dim:
            residual = nn.Dense(self.hidden_dim)(residual)
        return x + residual


class ResidualMLP(nn.Module):
    """Residual MLP with an initial projection, stacked residual blocks, and a final projection.

    Matches the ResidualMLP from the LARBRID PyTorch implementation.

    Attributes:
        hidden_dims: Hidden dimensions for the residual blocks.
        output_dim: Output dimension of the final linear layer.
    """

    hidden_dims: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dims[0])(x)
        dims = (self.hidden_dims[0],) + tuple(self.hidden_dims)
        for i in range(len(dims) - 1):
            x = ResidualMLPBlock(hidden_dim=dims[i + 1])(x)
        x = mish(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class PositionalFeature(nn.Module):
    """Sinusoidal positional embedding for scalar time inputs.

    Attributes:
        dim: Output embedding dimension (must be even).
        max_positions: Scaling factor for frequency computation.
    """

    dim: int
    max_positions: int = 10000

    @nn.compact
    def __call__(self, x):
        # x: (..., 1) float time index
        freqs = jnp.arange(0, self.dim // 2, dtype=jnp.float32)
        freqs = freqs / (self.dim // 2)
        freqs = (1.0 / self.max_positions) ** freqs  # (dim//2,)
        x = x * freqs  # (..., dim//2) via broadcasting
        return jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)  # (..., dim)


class NoiseFeature(nn.Module):
    """Maps (observation, noise) -> feature_dim, mirroring the DiffSR phi-network structure.

    Used as the bridging network in LARBRID: approximates phi(s, pi_dp(s, w)) from (s, w)
    so the noise actor can be updated without backpropagating through the flow policy.

    Attributes:
        embed_dim: Embedding dimension for obs and noise separately.
        feature_dim: Output feature dimension (same as DiffSR phi output).
        phi_hidden_dims: Hidden dimensions for the ResidualMLP backbone.
    """

    embed_dim: int
    feature_dim: int
    phi_hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, s, w):
        s_emb = nn.Dense(self.embed_dim * 2)(s)
        s_emb = mish(s_emb)
        s_emb = nn.Dense(self.embed_dim)(s_emb)

        w_emb = nn.Dense(self.embed_dim * 2)(w)
        w_emb = mish(w_emb)
        w_emb = nn.Dense(self.embed_dim)(w_emb)

        x = jnp.concatenate([s_emb, w_emb], axis=-1)
        x = ResidualMLP(hidden_dims=self.phi_hidden_dims, output_dim=self.feature_dim)(x)
        return x


class RffRewardNet(nn.Module):
    """Reward prediction network with learnable Random Fourier Features (RFF).

    Matches RffReward from the LARBRID PyTorch implementation.

    Attributes:
        rff_dim: Dimension of the RFF mapping (must be even).
        hidden_dims: Hidden dimensions for the MLP after RFF.
    """

    rff_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, phi):
        x = nn.LayerNorm()(phi)
        # Learnable RFF: project to rff_dim//2 then apply sin+cos
        x = nn.Dense(self.rff_dim // 2)(x)
        x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)  # (B, rff_dim)
        for d in self.hidden_dims:
            x = nn.Dense(d)(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)  # (B,)
