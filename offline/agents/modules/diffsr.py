"""DiffSR (Diffusion Spectral Representation) network modules for LARBRID.

These modules learn the low-rank MDP features via diffusion score matching
(Shribak et al., NeurIPS 2024). The phi network extracts state-action features
that linearly represent the transition kernel; the mu network captures the
corresponding next-state projection; the reward network predicts rewards from phi.

Reference: https://arxiv.org/abs/2406.16121
"""

from typing import Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import PositionalFeature, ResidualMLP, RffRewardNet, mish


class PhiNet(nn.Module):
    """Maps (s, a) -> feature_dim via separate embedding networks + ResidualMLP.

    Mirrors DDPM.mlp_s / mlp_a / mlp_phi from the LARBRID PyTorch implementation.

    Attributes:
        embed_dim: Embedding dimension for obs and action separately.
        feature_dim: Output feature dimension.
        hidden_dims: Hidden dimensions for the ResidualMLP backbone.
    """

    embed_dim: int
    feature_dim: int
    hidden_dims: Sequence[int]
    encoder: Optional[nn.Module] = None
    embed_scale: int = 2

    @nn.compact
    def __call__(self, s, a):
        if self.encoder is not None:
            s = self.encoder(s)

        s_emb = nn.Dense(self.embed_dim * self.embed_scale)(s)
        s_emb = mish(s_emb)
        s_emb = nn.Dense(self.embed_dim)(s_emb)

        a_emb = nn.Dense(self.embed_dim * self.embed_scale)(a)
        a_emb = mish(a_emb)
        a_emb = nn.Dense(self.embed_dim)(a_emb)

        x = jnp.concatenate([s_emb, a_emb], axis=-1)
        return ResidualMLP(hidden_dims=self.hidden_dims, output_dim=self.feature_dim)(x)


class MuNet(nn.Module):
    """Maps (noisy_next_state, time_index) -> (feature_dim, obs_dim) matrix.

    Mirrors DDPM.mlp_t / mlp_mu from the LARBRID PyTorch implementation.
    The epsilon prediction is: eps_pred = phi @ mu (batched matmul).

    Attributes:
        obs_dim: Observation space dimension.
        embed_dim: Embedding dimension for the time positional encoding.
        feature_dim: Feature dimension (must match PhiNet.feature_dim).
        hidden_dims: Hidden dimensions for the ResidualMLP backbone.
        final_tanh: Whether to apply tanh to the output before reshape, bounding
            mu to [-1, 1] and stabilising the score-matching scale (matches the
            DiffSR visual reference implementation).
    """

    obs_dim: int
    embed_dim: int
    feature_dim: int
    hidden_dims: Sequence[int]
    final_tanh: bool = False
    encoder: Optional[nn.Module] = None
    embed_scale: int = 2

    @nn.compact
    def __call__(self, sp, t):
        if self.encoder is not None:
            sp = self.encoder(sp)

        # t: (B, 1) float time index
        t_emb = PositionalFeature(dim=self.embed_dim)(t)   # (B, embed_dim)
        t_emb = nn.Dense(self.embed_dim * self.embed_scale)(t_emb)
        t_emb = mish(t_emb)
        t_emb = nn.Dense(self.embed_dim)(t_emb)

        x = jnp.concatenate([sp, t_emb], axis=-1)
        x = ResidualMLP(
            hidden_dims=self.hidden_dims,
            output_dim=self.feature_dim * self.obs_dim,
        )(x)
        if self.final_tanh:
            x = jnp.tanh(x)
        # Reshape to (B, feature_dim, obs_dim) for batched matmul with phi
        return x.reshape(-1, self.feature_dim, self.obs_dim)
