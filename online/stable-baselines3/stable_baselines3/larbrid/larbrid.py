from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import torch.nn as nn

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousFeatureCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor
from stable_baselines3.larbrid.policies import LARBRIDPolicy, MlpPolicy, CnnPolicy, MultiInputPolicy

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

SelfLARBRID = TypeVar("SelfLARBRID", bound="LARBRID")

class LARBRID(OffPolicyAlgorithm):
    """
    LARBRID.
    Uses a SAC-style stochastic actor to generate noises for a frozen pre-trained diffusion policy.
    The DDPM module learns low-rank MDP features; the critic operates on these features.
    A noise_feature network distils the DDPM features into the noise domain so the actor
    can be updated without passing gradients through the diffusion model.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from
    :param feature_space: Feature space of the DDPM feature extractor
    :param learning_rate: Learning rate for actor, critic, and noise_feature optimizers
    :param buffer_size: Replay buffer size
    :param learning_starts: Steps collected before learning starts
    :param batch_size: Minibatch size for gradient updates
    :param tau: Polyak averaging rate for target networks
    :param gamma: Discount factor
    :param train_freq: Environment steps between updates
    :param gradient_steps: Gradient steps per update (UTD ratio)
    :param action_noise: Optional action noise for exploration
    :param ent_coef: Entropy coefficient ('auto' to learn it automatically)
    :param target_entropy: Target entropy for automatic ent_coef tuning
    :param target_update_interval: Critic target update frequency (in gradient steps)
    :param actor_gradient_steps: Actor update frequency; -1 means every gradient step
    :param noise_feature_grad_steps: Distillation steps for noise_feature per train() call
    :param diffusion_policy: Frozen pre-trained diffusion policy
    :param diffusion_act_dim: (action_chunk_length, action_dim) of the diffusion policy
    :param critic_backup_combine_type: 'min' or 'mean' over critic ensemble for backup
    :param ddpm_lr: Optional separate learning rate for DDPM optimizer
    """
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: LARBRIDPolicy
    actor: Actor
    critic: ContinuousFeatureCritic
    critic_target: ContinuousFeatureCritic

    def __init__(
        self,
        policy: Union[str, type[LARBRIDPolicy]],
        env: Union[GymEnv, str],
        feature_space: spaces,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        target_update_interval: int = 1,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        actor_gradient_steps: int = -1,
        noise_feature_grad_steps: int = 1,
        diffusion_policy=None,
        diffusion_act_dim=None,
        critic_backup_combine_type='min',
        ddpm_lr: Optional[float] = None,
        max_grad_norm: float = 10.0,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
            feature_space=feature_space,
        )

        self.target_update_interval = target_update_interval
        self.actor_gradient_steps = actor_gradient_steps
        self.noise_feature_grad_steps = noise_feature_grad_steps
        self.diffusion_policy = diffusion_policy
        self.diffusion_act_chunk = diffusion_act_dim[0]
        self.diffusion_act_dim = diffusion_act_dim[1]
        self.critic_backup_combine_type = critic_backup_combine_type
        self.max_grad_norm = max_grad_norm

        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.log_ent_coef: Optional[th.Tensor] = None
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        if ddpm_lr is not None:
            if self.policy_kwargs is None:
                self.policy_kwargs = {}
            self.policy_kwargs["ddpm_lr"] = ddpm_lr

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

        # Target entropy: default to -|action_dim| if auto
        if self.target_entropy == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))
        else:
            self.target_entropy = float(self.target_entropy)

        # Entropy coefficient setup (SAC-style)
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.ddpm = self.policy.ddpm
        self.ddpm_target = self.policy.ddpm_target
        self.noise_feature = self.policy.noise_feature

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)

        # update optimisers for actor, critic and nosie_feature together as they use the same lr
        # but ddpm use different lr that is directly passed into it.
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.noise_feature.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, diffusion_losses, reward_losses, noise_feat_losses = [], [], [], [], []

        if self.actor_gradient_steps < 0:
            actor_gradient_idx = np.linspace(0, gradient_steps-1, gradient_steps, dtype=int)
        else:
            actor_gradient_idx = np.linspace(int(gradient_steps / self.actor_gradient_steps) - 1, gradient_steps-1, self.actor_gradient_steps, dtype=int)

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Sample noise actions and log probs from the SAC actor for current obs
            # (used for entropy coef update and actor loss)
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # Entropy coefficient update
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                # log_prob is detached so ent_coef_loss only affects log_ent_coef
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # DDPM loss: learns transition kernel and reward function
            _, diffusion_loss, reward_loss = self.ddpm.compute_loss(
                replay_data.observations,
                replay_data.actions,
                replay_data.next_observations,
                replay_data.rewards,
            )
            feat_losses = diffusion_loss + reward_loss
            diffusion_losses.append(diffusion_loss.item())
            reward_losses.append(reward_loss.item())

            # Critic loss (SAC-style Bellman backup with entropy term, using DDPM features)
            feature = self.get_feature(replay_data.observations, replay_data.actions).detach()
            with th.no_grad():
                # Next noise actions from SAC actor (stochastic)
                next_actions, next_log_prob = self.actor_target.action_log_prob(replay_data.next_observations)
                # Unscale from [-1,1] to action_space bounds, then diffuse
                next_actions = th.tensor(self.policy.unscale_action(next_actions.cpu().numpy())).to(self.device)
                next_actions = self.diffusion_policy(
                    replay_data.next_observations,
                    next_actions.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim),
                    return_numpy=False,
                )
                next_actions = next_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)

                # Get features and target Q-values
                next_feat = self.ddpm_target.compute_feature(replay_data.next_observations, next_actions)
                next_q_values = th.cat(self.critic_target(next_feat), dim=1)
                if self.critic_backup_combine_type == 'min':
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                elif self.critic_backup_combine_type == 'mean':
                    next_q_values = th.mean(next_q_values, dim=1, keepdim=True)

                # Add entropy term to target (SAC)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(feature)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Step DDPM and critic
            self.ddpm.optimizer.zero_grad()
            feat_losses.backward()
            th.nn.utils.clip_grad_norm_(self.ddpm.parameters(), self.max_grad_norm)
            self.ddpm.optimizer.step()

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic.optimizer.step()

            # feature bridging before learning actor, we make the noise feature aims to match target feature
            for _ in range(self.noise_feature_grad_steps):
                feature_distill_loss = 0
                distill_replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                feature_distill_loss += self.update_noise_feature(distill_replay_data)
                noise_feat_losses.append(feature_distill_loss.item())
                self.noise_feature.optimizer.zero_grad()
                feature_distill_loss.backward()
                th.nn.utils.clip_grad_norm_(self.noise_feature.parameters(), self.max_grad_norm)
                self.noise_feature.optimizer.step()

            # Actor loss: SAC entropy + LARBRID feature bridge (noise_feature -> critic)
            # Gradients flow: actor -> actions_pi -> noise_feature -> actor_feat -> critic(forward) -> sac loss
            if gradient_step in actor_gradient_idx:
                actor_feat = self.noise_feature(replay_data.observations, actions_pi)

                # check the noise feature output matches original feature
                # with th.no_grad():
                #     diffused_actions = th.tensor(self.policy.unscale_action(actions_pi.cpu().numpy())).to(self.device)
                #     diffused_actions = self.diffusion_policy(replay_data.observations, 
                #                                              diffused_actions.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim),
                #                                              return_numpy=False)
                #     diffused_actions = diffused_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
                #     target_feat = self.ddpm.compute_feature(replay_data.observations, diffused_actions)
                #     mismatch = th.linalg.norm(actor_feat - target_feat).cpu().numpy()
                #     feat_mismatch.append(mismatch)

                q_values_pi = th.cat(self.critic(actor_feat), dim=1)
                if self.critic_backup_combine_type == 'min':
                    min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                elif self.critic_backup_combine_type == 'mean':
                    min_qf_pi = th.mean(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.ddpm.parameters(), self.ddpm_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/diffusion_loss", np.mean(diffusion_losses))
        self.logger.record("train/reward_loss", np.mean(reward_losses))
        self.logger.record("train/noise_feat_loss", np.mean(noise_feat_losses))
        # self.logger.record("train/feat_miss_match", np.mean(feat_mismatch))
        # self.logger.record("train/feat_miss_match_max", np.max(feat_mismatch))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfLARBRID,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "LARBRID",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfLARBRID:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def get_feature(self, observations, actions, use_target=True):
        model = self.ddpm_target if use_target else self.ddpm
        return model.compute_feature(observations, actions)

    # def update_noise_feature(self, replay_data):
    #     """Distil noise_feature to approximate phi_tilde(s,w) = phi(s, pi_dp(s,w)).
    #     Samples random noises, diffuses them, computes ddpm_target features as targets,
    #     then trains noise_feature via MSE.
    #     The random noises are passed through tanh to match the SAC actor's output distribution.
    #     """
    #     with th.no_grad():
    #         noise = th.randn(replay_data.actions.shape[0], self.diffusion_act_chunk, self.diffusion_act_dim).to(self.device)
    #         diffused_actions = self.diffusion_policy(replay_data.observations, noise, return_numpy=False)
    #         diffused_actions = diffused_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
    #         target_feat = self.ddpm_target.compute_feature(replay_data.observations, diffused_actions)
    #     # Apply tanh to match the SAC actor's squashed output distribution in (-1, 1)
    #     noise_scaled = noise.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim).detach().cpu().numpy()
    #     noise_scaled = th.tensor(self.policy.scale_action(noise_scaled)).to(self.device)
    #     pred_feat = self.noise_feature(replay_data.observations, noise_scaled)
    #     return F.mse_loss(pred_feat, target_feat)
    
    def update_noise_feature(self, replay_data):
        """Distil noise_feature to approximate phi_tilde(s,w) = phi(s, pi_dp(s,w)).
        Samples noises from the actor_target to match the inference distribution (tanh-squashed),
        diffuses them, computes ddpm action features as targets, then trains noise_feature via MSE.
        """
        with th.no_grad():
            # Sample from actor_target so the training distribution matches inference (tanh-squashed outputs)
            noise_scaled, _ = self.actor_target.action_log_prob(replay_data.observations)
            # Unscale from [-1,1] to action-space bounds for the diffusion policy
            noise_unscaled = th.tensor(self.policy.unscale_action(noise_scaled.cpu().numpy())).to(self.device)
            diffused_actions = self.diffusion_policy(
                replay_data.observations,
                noise_unscaled.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim),
                return_numpy=False,
            )
            diffused_actions = diffused_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
            target_feat = self.ddpm.compute_feature(replay_data.observations, diffused_actions)
        pred_feat = self.noise_feature(replay_data.observations, noise_scaled.detach())
        return F.mse_loss(pred_feat, target_feat)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target", "ddpm_target", "actor_target"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "noise_feature", "noise_feature.optimizer"]
        return state_dicts, []

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps < learning_starts:
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            buffer_action = unscaled_action
            action = buffer_action

        action = th.as_tensor(action, device=self.device, dtype=th.float32)
        obs = th.as_tensor(self._last_obs, device=self.device, dtype=th.float32)
        action = self.diffusion_policy(obs, action.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim), return_numpy=False)
        action = action.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
        action = action.cpu().numpy()
        buffer_action = action
        return action, buffer_action

    def predict_diffused(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        unscaled_action, predict_second_return = self.policy.predict(observation, state, episode_start, deterministic)
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            buffer_action = unscaled_action
            action = buffer_action
        action = th.as_tensor(action, device=self.device, dtype=th.float32)
        obs = th.as_tensor(observation, device=self.device, dtype=th.float32)
        action = self.diffusion_policy(obs, action.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim), return_numpy=False)
        action = action.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
        action = action.cpu().numpy()
        return action, predict_second_return
