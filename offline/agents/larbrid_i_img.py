"""LARBRID-I with Image Observations (LARBRID-I-Img).

Follows Appendix F of DiffSR (Shribak et al., NeurIPS 2024).  Rather than
diffusing in raw pixel space, we first encode each frame with a shared
convolutional AE, then run the DiffSR score-matching loss in the compact
latent space.

Architecture (single TrainState, flat ModuleDict):
  - encoder:            VisualEncoder  (B, H, W, C) → (B, latent_dim)
  - target_encoder:     EMA copy, used for next-state targets
  - decoder:            VisualDecoder  (B, latent_dim) → (B, H, W, C)
  - diffsr_phi:         PhiNet  (state_latent, a) → (B, F)
  - diffsr_mu:          MuNet   (noisy_next_latent, t) → (B, F, L)
  - diffsr_reward:      RffRewardNet  phi → scalar
  - critic:             Value   phi → (2, B)
  - actor_bc_flow:      ActorVectorField  (latent, a, t) → velocity
  - actor_onestep_flow: ActorVectorField  (latent, noise) → action
  - target_critic:      EMA copy of critic

Gradient flow (matches DiffSR visual agent):
  - ae_loss      → encoder + decoder only
  - diffsr_loss  → phi + mu + reward only  (encoder frozen, like DiffSR no_grad)
  - critic_loss  → critic only             (encoder + phi frozen, like phi.detach())
  - actor_loss   → actor (bc_flow, onestep_flow) only
                   (encoder/phi/critic frozen, like DiffSR latent.detach() +
                    actor_optim.step() only)

Target encoder/phi/critic are updated by slow Polyak averaging; they
provide stable TD targets and are never trained directly by loss gradients.

Warmup: call pretrain_ae() for the first N steps to warm up the AE before
the full RL training loop begins (handled in main.py).
"""

import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.modules.diffsr import MuNet, PhiNet
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, RffRewardNet, Value


# ---------------------------------------------------------------------------
# Image augmentation
# ---------------------------------------------------------------------------

def _random_shift_aug(x, pad, rng):
    """Random shift augmentation for NHWC image batches.

    Pads each image by `pad` pixels on all sides using edge-replicate padding,
    then randomly crops back to the original H×W with a per-sample offset.
    Matches the RandomShiftsAug used in DiffSR / DrQ-v2.

    Args:
        x:   (B, H, W, C) image array (uint8 or float32)
        pad: int, number of pixels to pad on each side
        rng: JAX PRNG key
    Returns: (B, H, W, C) augmented image, same dtype as x
    """
    B, H, W, C = x.shape
    x_pad = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='edge')
    rng_h, rng_w = jax.random.split(rng)
    h_shifts = jax.random.randint(rng_h, (B,), 0, 2 * pad + 1)
    w_shifts = jax.random.randint(rng_w, (B,), 0, 2 * pad + 1)

    def _crop(args):
        img, hs, ws = args
        return jax.lax.dynamic_slice(img, (hs, ws, 0), (H, W, C))

    return jax.vmap(_crop)((x_pad, h_shifts, w_shifts))


# ---------------------------------------------------------------------------
# Visual encoder / decoder
# ---------------------------------------------------------------------------

class VisualEncoder(nn.Module):
    """Convolutional encoder: (B, H, W, C) → (B, latent_dim).

    Uses SAME padding so the encoded spatial size is exactly ceil(H/2) ×
    ceil(W/2), which the decoder can invert exactly with a stride-2
    ConvTranspose.

    Normalises input pixels to [-0.5, 0.5] internally.
    """

    latent_dim: int
    num_filters: int = 32
    num_layers: int = 4

    @nn.compact
    def __call__(self, obs):
        x = obs.astype(jnp.float32) / 255.0 - 0.5
        x = nn.Conv(self.num_filters, (3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        for _ in range(self.num_layers - 1):
            x = nn.Conv(self.num_filters, (3, 3), strides=(1, 1), padding='SAME')(x)
            x = nn.relu(x)
        x = x.reshape((*obs.shape[:-3], -1))
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = jax.nn.swish(x)   # swish
        x = nn.Dense(self.latent_dim)(x)
        return x


class VisualDecoder(nn.Module):
    """Deconvolutional decoder: (B, latent_dim) → (B, H, W, out_channels).

    Mirrors VisualEncoder exactly (SAME padding throughout).  The caller
    must provide the encoded spatial size (H_enc, W_enc) computed as
    ceil(H/2), ceil(W/2) for image size H × W.
    """

    enc_flat_dim: int    # H_enc * W_enc * num_filters
    enc_h: int           # ceil(H / 2)
    enc_w: int           # ceil(W / 2)
    out_channels: int    # C per reconstructed frame
    num_filters: int = 32
    num_layers: int = 4

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(self.enc_flat_dim)(z)
        x = nn.relu(x)
        x = x.reshape((*z.shape[:-1], self.enc_h, self.enc_w, self.num_filters))
        for _ in range(self.num_layers - 1):
            x = nn.ConvTranspose(self.num_filters, (3, 3), strides=(1, 1), padding='SAME')(x)
            x = nn.relu(x)
        x = nn.ConvTranspose(self.out_channels, (3, 3), strides=(2, 2), padding='SAME')(x)
        return x  # (B, H, W, out_channels), unbounded — target is in [-0.5, 0.5]


# ---------------------------------------------------------------------------
# VP noise schedule helper (unchanged from larbrid_i.py)
# ---------------------------------------------------------------------------

def _vp_alphabars(sample_steps: int):
    t = jnp.arange(1, sample_steps + 1, dtype=jnp.float32)
    b_max, b_min = 10.0, 0.1
    alpha = jnp.exp(-b_min / sample_steps - 0.5 * (b_max - b_min) * (2 * t - 1) / sample_steps ** 2)
    return jnp.cumprod(alpha)   # (T,)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class LARBRIDIImgAgent(flax.struct.PyTreeNode):
    """LARBRID-I with a shared visual encoder and DiffSR in latent space."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Internal encoding helper
    # ------------------------------------------------------------------

    def _encode_frames(self, obs, encoder_key, params=None):
        """Encode stacked observations to a flat latent vector.

        Args:
            obs:         (B, H, W, C * frame_stack)
            encoder_key: 'encoder' or 'target_encoder'
            params:      grad_params to use (None → current network params)

        Returns:
            (B, latent_dim * frame_stack)
        """
        fs = self.config['frame_stack']
        C = self.config['channels_per_frame']
        B = obs.shape[0]
        H, W = obs.shape[1], obs.shape[2]
        # Correct split: isolate the frame axis within channels first, then
        # move it to the batch position.
        # (B, H, W, C*fs) → (B, H, W, fs, C) → (B, fs, H, W, C) → (B*fs, H, W, C)
        obs_frames = obs.reshape(B, H, W, fs, C)
        obs_frames = jnp.transpose(obs_frames, (0, 3, 1, 2, 4))
        obs_frames = obs_frames.reshape(B * fs, H, W, C)
        latents = self.network.select(encoder_key)(obs_frames, params=params)
        return latents.reshape(B, -1)   # (B, latent_dim * fs)

    # ------------------------------------------------------------------
    # AE loss: reconstruction + variance collapse prevention
    # ------------------------------------------------------------------

    def ae_loss(self, batch, grad_params, rng):
        """MSE reconstruction loss + variance regularisation on the encoder."""
        next_obs = batch['next_observations']   # (B, H, W, C*fs)
        C = self.config['channels_per_frame']
        # Reconstruct the last frame of the next observation, matching the
        # visual DiffSR ae_step which uses next_img_step as both input and target.
        last_frame = next_obs[..., -C:]                       # (B, H, W, C)
        target = last_frame.astype(jnp.float32) / 255.0 - 0.5

        # Apply random shift augmentation to the encoder input only; the
        # reconstruction target remains the unaugmented frame.
        aug_frame = _random_shift_aug(last_frame, self.config['ae_aug_pad'], rng)

        # Encode the augmented last frame with the *trainable* encoder.
        z = self.network.select('encoder')(aug_frame, params=grad_params)
        recon = self.network.select('decoder')(z, params=grad_params)

        # Mirror DiffSR ae_step: sum over all elements, divide by batch size.
        # Equivalent to mean_over_B(sum_over_H_W_C(squared_error)).
        recon_loss = jnp.sum((recon - target) ** 2) / recon.shape[0]
        # Variance collapse penalty: matches DiffSR (1 - std).clip(0).sum().
        latent_std = z.std(axis=0)
        var_loss = jnp.sum(jnp.clip(1.0 - latent_std, 0.0))

        total = self.config['ae_recon_coef'] * recon_loss + self.config['ae_var_coef'] * var_loss
        return total, {
            'ae/recon_loss': recon_loss,
            'ae/var_loss': var_loss,
            'ae/total_loss': total,
            'ae/latent_mean': z.mean(),
            'ae/latent_std': latent_std.mean(),
        }

    # ------------------------------------------------------------------
    # DiffSR loss (score matching in latent space)
    # ------------------------------------------------------------------

    def diffsr_loss(self, batch, grad_params, rng):
        """DiffSR score matching + reward prediction, operating in latent space."""
        s   = batch['observations']
        a   = batch['actions']
        sp  = batch['next_observations']
        r   = batch['rewards']

        rng, t_rng, eps_rng = jax.random.split(rng, 3)
        T = self.config['diffsr_sample_steps']
        t_idx = jax.random.randint(t_rng, (s.shape[0],), 0, T)

        s_latent = jax.lax.stop_gradient(
            self._encode_frames(s, 'encoder')
        )
        sp_latent = jax.lax.stop_gradient(
            self._encode_frames(sp, 'target_encoder')
        )

        alphabars = _vp_alphabars(T)
        ab = alphabars[t_idx]        # (B,)
        eps = jax.random.normal(eps_rng, sp_latent.shape)
        sp_noisy = jnp.sqrt(ab[:, None]) * sp_latent + jnp.sqrt(1.0 - ab[:, None]) * eps

        phi = self.network.select('diffsr_phi')(
            s_latent, a, params=grad_params
        )
        mu = self.network.select('diffsr_mu')(
            sp_noisy, t_idx[:, None].astype(jnp.float32), params=grad_params
        )

        eps_pred = jnp.einsum('bf,bfo->bo', phi, mu)
        diffusion_loss = jnp.mean(jnp.sum((eps_pred - eps) ** 2, axis=-1))

        reward_pred  = self.network.select('diffsr_reward')(phi, params=grad_params)
        reward_loss  = jnp.mean((reward_pred - r) ** 2)

        total = diffusion_loss + reward_loss
        return total, {
            'diffsr/total_loss':     total,
            'diffsr/diffusion_loss': diffusion_loss,
            'diffsr/reward_loss':    reward_loss,
            'diffsr/phi_norm':       jnp.mean(jnp.linalg.norm(phi, axis=-1)),
        }

    # ------------------------------------------------------------------
    # Critic loss (TD backup in phi space)
    # ------------------------------------------------------------------

    def critic_loss(self, batch, grad_params, rng):
        """TD critic loss.  Only critic receives gradients."""
        rng, sample_rng = jax.random.split(rng)

        # --- TD target ---------------------------------------------------
        # Encode next observations with target encoder (stable, no grad).
        sp_latent_tgt = jax.lax.stop_gradient(
            self._encode_frames(batch['next_observations'], 'target_encoder') # default was: target_encoder
        )
        # Sample next actions using the target-encoded next state.
        next_actions = self._sample_actions_from_latent(sp_latent_tgt, seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_phi  = self.network.select('diffsr_phi')(sp_latent_tgt, next_actions) # default was: diffsr_phi
        next_qs   = self.network.select('target_critic')(next_phi)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        # --- Current Q: current encoder + target phi, only critic updated --------
        s_latent    = jax.lax.stop_gradient(
            self._encode_frames(batch['observations'], 'encoder') # default was encoder
        )
        current_phi = self.network.select('diffsr_phi')(s_latent, batch['actions']) # default was: diffsr_phi
        q = self.network.select('critic')(current_phi, params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic/critic_loss': critic_loss,
            'critic/q_mean': q.mean(),
            'critic/q_max':  q.max(),
            'critic/q_min':  q.min(),
        }

    # ------------------------------------------------------------------
    # Actor loss (BC flow + distillation + Q)
    # ------------------------------------------------------------------

    def actor_loss(self, batch, grad_params, rng):
        """FQL actor loss.  Only actor params (bc_flow, onestep_flow) are updated.

        Target encoder and target phi are both frozen; Q-gradient reaches the
        actor through actor_actions via the chain rule through target_diffsr_phi.
        """
        batch_size  = batch['actions'].shape[0]
        action_dim  = self.config['action_dim']

        s_encoded = jax.lax.stop_gradient(
            self._encode_frames(batch['observations'], 'encoder') # default was encoder
        )

        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # --- BC flow loss ------------------------------------------------
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t   = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(
            s_encoded, x_t, t, params=grad_params
        )
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # --- Distillation loss -------------------------------------------
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))

        # s_encoded already has no gradient path to encoder, so teacher and
        # student both use the same frozen latent as input.
        target_flow_actions = self._flow_euler_from_latent(s_encoded, noises)

        actor_actions = self.network.select('actor_onestep_flow')(
            s_encoded, noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # --- Q loss: target phi weights frozen, gradient reaches actor via actions
        actor_actions_clipped = jnp.clip(actor_actions, -1, 1)
        actor_phi = self.network.select('diffsr_phi')(s_encoded, actor_actions_clipped) # default was diffsr_phi
        qs = self.network.select('critic')(actor_phi)
        q  = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam    = jax.lax.stop_gradient(1.0 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        actions = self.sample_actions(batch['observations'], seed=rng)
        mse     = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor/actor_loss':    actor_loss,
            'actor/bc_flow_loss':  bc_flow_loss,
            'actor/distill_loss':  distill_loss,
            'actor/q_loss':        q_loss,
            'actor/q':             q.mean(),
            'actor/mse':           mse,
        }

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, ae_rng, diffsr_rng, critic_rng, actor_rng = jax.random.split(rng, 5)

        ae_loss,     ae_info     = self.ae_loss(batch, grad_params, ae_rng)
        diffsr_loss, diffsr_info = self.diffsr_loss(batch, grad_params, diffsr_rng)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        actor_loss,  actor_info  = self.actor_loss(batch, grad_params, actor_rng)

        for k, v in ae_info.items():     info[k] = v
        for k, v in diffsr_info.items(): info[k] = v
        for k, v in critic_info.items(): info[k] = v
        for k, v in actor_info.items():  info[k] = v

        loss = ae_loss + diffsr_loss + critic_loss + actor_loss
        return loss, info

    # ------------------------------------------------------------------
    # Polyak (EMA) target update
    # ------------------------------------------------------------------

    def target_update(self, network, module_name, tau=None):
        if tau is None:
            tau = self.config['tau']
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target

    # ------------------------------------------------------------------
    # Main update step
    # ------------------------------------------------------------------

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        self.target_update(new_network, 'critic')
        self.target_update(new_network, 'diffsr_phi')
        self.target_update(new_network, 'encoder',
                           tau=self.config['encoder_tau'])

        return self.replace(network=new_network, rng=new_rng), info

    # ------------------------------------------------------------------
    # AE warm-up step (encoder + decoder only)
    # ------------------------------------------------------------------

    @jax.jit
    def pretrain_ae(self, batch):
        """One gradient step on the AE loss only.

        All other parameters have zero gradient so they are unchanged.
        The target encoder is fully synced to the live encoder after each
        warm-up step (tau = 1), consistent with the DiffSR visual reference.
        """
        new_rng, ae_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            loss, info = self.ae_loss(batch, grad_params, ae_rng)
            return loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        # Full copy: target encoder ← live encoder during warm-up.
        new_network.params['modules_target_encoder'] = (
            new_network.params['modules_encoder']
        )
        return self.replace(network=new_network, rng=new_rng), info

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    @jax.jit
    def reconstruct(self, observations):
        """Encode and decode the last frame of observations for visualization.

        Returns (input_imgs, recon_imgs) as uint8 (B, H, W, C) arrays.
        """
        C = self.config['channels_per_frame']
        last_frame = observations[..., -C:]   # (B, H, W, C)
        z = self.network.select('encoder')(last_frame)
        recon = self.network.select('decoder')(z)
        recon_uint8 = jnp.clip((recon + 0.5) * 255.0, 0, 255).astype(jnp.uint8)
        return last_frame, recon_uint8

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Sample actions from the one-step flow policy."""
        action_seed, _ = jax.random.split(seed)
        unbatched = observations.ndim == 3
        if unbatched:
            observations = observations[None]
        encoded = self._encode_frames(observations, 'encoder')
        noises  = jax.random.normal(
            action_seed,
            (*observations.shape[:-3], self.config['action_dim']),
        )
        actions = self.network.select('actor_onestep_flow')(encoded, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions[0] if unbatched else actions

    # ------------------------------------------------------------------
    # Internal helpers (not @jit — called inside jit'd methods)
    # ------------------------------------------------------------------

    def _sample_actions_from_latent(self, encoded_obs, seed):
        """Sample one-step flow actions from pre-encoded observations."""
        action_seed, _ = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (*encoded_obs.shape[:-1], self.config['action_dim']),
        )
        actions = self.network.select('actor_onestep_flow')(encoded_obs, noises)
        return jnp.clip(actions, -1, 1)

    def _flow_euler_from_latent(self, encoded_obs, noises):
        """Euler integration of the BC flow from pre-encoded observations."""
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*encoded_obs.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(encoded_obs, actions, t)
            actions = actions + vels / self.config['flow_steps']
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        """Public interface: Euler BC flow from raw image observations."""
        encoded = self._encode_frames(observations, 'encoder')
        return self._flow_euler_from_latent(encoded, noises)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # ---- Infer image geometry ----------------------------------------
        H, W, C_total = (
            ex_observations.shape[-3],
            ex_observations.shape[-2],
            ex_observations.shape[-1],
        )
        channels_per_frame = config.get('channels_per_frame', 3)
        frame_stack = C_total // channels_per_frame

        # Encoded spatial dims (SAME-padded stride-2 first conv).
        H_enc = (H + 1) // 2
        W_enc = (W + 1) // 2
        ae_num_filters = config.get('ae_num_filters', 32)
        ae_enc_flat    = H_enc * W_enc * ae_num_filters
        ae_latent_dim  = config.get('ae_latent_dim', 50)
        state_latent_dim = ae_latent_dim * frame_stack   # concatenated latent

        action_dim = ex_actions.shape[-1]
        ob_dims    = ex_observations.shape[1:]

        # ---- Example tensors ---------------------------------------------
        ex_single_frame = ex_observations[..., :channels_per_frame]  # (B, H, W, C)
        ex_latent_single = jnp.zeros((ex_observations.shape[0], ae_latent_dim))
        ex_state_latent  = jnp.zeros((ex_observations.shape[0], state_latent_dim))
        ex_times         = ex_actions[..., :1]
        ex_phi           = jnp.zeros((ex_observations.shape[0], config['diffsr_feature_dim']))
        ex_t_diffsr      = jnp.zeros((ex_observations.shape[0], 1))

        # ---- Network definitions -----------------------------------------
        encoder_def = VisualEncoder(
            latent_dim=ae_latent_dim,
            num_filters=ae_num_filters,
            num_layers=config.get('ae_num_layers', 4),
        )
        decoder_def = VisualDecoder(
            enc_flat_dim=ae_enc_flat,
            enc_h=H_enc,
            enc_w=W_enc,
            out_channels=channels_per_frame,
            num_filters=ae_num_filters,
            num_layers=config.get('ae_num_layers', 4),
        )

        phi_def = PhiNet(
            embed_dim=config['diffsr_embed_dim'],
            feature_dim=config['diffsr_feature_dim'],
            hidden_dims=config['diffsr_phi_hidden_dims'],
            embed_scale=2,
        )
        mu_def = MuNet(
            obs_dim=state_latent_dim,
            embed_dim=config['diffsr_embed_dim'],
            feature_dim=config['diffsr_feature_dim'],
            hidden_dims=config['diffsr_mu_hidden_dims'],
            final_tanh=True,
            embed_scale=2,
        )
        reward_def = RffRewardNet(
            rff_dim=config['diffsr_rff_dim'],
            hidden_dims=config['diffsr_reward_hidden_dims'],
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        # Actors receive pre-encoded observations — no internal encoder.
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=None,
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=None,
        )

        network_info = dict(
            encoder               = (encoder_def,                    (ex_single_frame,)),
            target_encoder        = (copy.deepcopy(encoder_def),     (ex_single_frame,)),
            decoder               = (decoder_def,                    (ex_latent_single,)),
            diffsr_phi            = (phi_def,                        (ex_state_latent, ex_actions)),
            target_diffsr_phi     = (copy.deepcopy(phi_def),         (ex_state_latent, ex_actions)),
            diffsr_mu             = (mu_def,                         (ex_state_latent, ex_t_diffsr)),
            diffsr_reward         = (reward_def,                     (ex_phi,)),
            critic                = (critic_def,                     (ex_phi,)),
            target_critic         = (copy.deepcopy(critic_def),      (ex_phi,)),
            actor_bc_flow         = (actor_bc_flow_def,              (ex_state_latent, ex_actions, ex_times)),
            actor_onestep_flow    = (actor_onestep_flow_def,         (ex_state_latent, ex_actions)),
        )

        networks    = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        # ---- Optimizer (three learning-rate groups) ----------------------
        _ae_keys = {'modules_encoder', 'modules_decoder'}
        _diffsr_keys = {
            'modules_diffsr_phi',
            'modules_diffsr_mu',
            'modules_diffsr_reward',
        }

        def param_labels(params):
            return {
                k: jax.tree_util.tree_map(
                    lambda _: (
                        'ae'     if k in _ae_keys     else
                        'diffsr' if k in _diffsr_keys else
                        'main'
                    ),
                    v,
                )
                for k, v in params.items()
            }

        network_tx = optax.multi_transform(
            {
                'ae':     optax.adam(learning_rate=config.get('ae_lr', 3e-4)),
                'diffsr': optax.adam(learning_rate=config['diffsr_lr']),
                'main':   optax.adam(learning_rate=config['lr']),
            },
            param_labels=param_labels,
        )

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Initialise all target networks as exact copies of their live peers.
        p = network.params
        p['modules_target_encoder']      = p['modules_encoder']
        p['modules_target_critic']       = p['modules_critic']
        p['modules_target_diffsr_phi']   = p['modules_diffsr_phi']

        # Store computed geometry in config so agent methods can access it.
        config['ob_dims']             = ob_dims
        config['action_dim']          = action_dim
        config['frame_stack']         = frame_stack
        config['channels_per_frame']  = channels_per_frame
        config['ae_latent_dim']       = ae_latent_dim
        config['state_latent_dim']    = state_latent_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='larbrid_i_img',
        ob_dims=ml_collections.config_dict.placeholder(list),
        action_dim=ml_collections.config_dict.placeholder(int),

        # Shared
        lr=3e-4,
        batch_size=256,
        discount=0.99,
        tau=0.01,
        q_agg='mean',

        # Actor
        actor_hidden_dims=(512, 512, 512,),
        actor_layer_norm=False,
        alpha=10.0,
        flow_steps=10,
        normalize_q_loss=False,

        # Critic (operates on phi features)
        value_hidden_dims=(512, 512, 512),
        layer_norm=True,

        # Visual AE
        ae_latent_dim=128,
        ae_num_filters=32,
        ae_num_layers=4,
        ae_lr=1e-4,
        ae_recon_coef=1,
        ae_var_coef=0.1,
        ae_aug_pad=4,        # random-shift padding pixels (DrQ-v2 / DiffSR style)
        encoder_tau=0.01,    # slow EMA rate for target encoder
        channels_per_frame=3,

        # AE warm-up
        ae_warmup_steps=20000,

        # DiffSR
        diffsr_lr=1e-4,
        diffsr_embed_dim=256,
        diffsr_feature_dim=512,
        diffsr_phi_hidden_dims=(1024, ),
        diffsr_mu_hidden_dims=(1024, ),
        diffsr_reward_hidden_dims=(1024, ),
        diffsr_rff_dim=1024,
        diffsr_sample_steps=50,
    ))
    return config
