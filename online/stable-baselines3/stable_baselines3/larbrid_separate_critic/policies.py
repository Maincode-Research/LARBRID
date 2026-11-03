from typing import Any, Optional, Union

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, ContinuousFeatureCritic
from stable_baselines3.sac.policies import Actor
from stable_baselines3.larbrid.module.ddpm import DDPM
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)

from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


class LARBRIDSeparateCriticPolicy(BasePolicy):
    """
    Policy class (with both actor and critics) for the LARBRID "separate critic" ablation.

    Unlike LARBRIDPolicy, this variant does not distil a noise_feature network to bridge
    the noise space into the DDPM feature space. Instead it keeps two independent
    critics:
      - ``critic`` / ``critic_target``: a ContinuousFeatureCritic operating on DDPM
        features phi(s, a) of the diffused action, exactly as in LARBRID. This is what
        keeps the low-rank-MDP representation learning objective in the loop.
      - ``critic_noise``: a plain ContinuousCritic Q(s, w) that consumes the raw
        (observation, noise) pair directly, mirroring DSRL's critic_noise. It has no
        target network of its own; it is trained by regression (MSE) onto the action
        critic's values, and the actor is updated directly against it.

    :param observation_space: Observation space
    :param action_space: Action space (this is the noise space fed to the frozen diffusion policy)
    :param feature_space: Feature space output by DDPM
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create (for both the action critic
        and the noise critic).
    :param log_std_init: Initial value for the log standard deviation of the SAC actor.
    :param ddpm_lr: Optional separate learning rate for the DDPM optimizer.
    :param ddpm_kwargs: Arguments forwarded to DDPM constructor.
    """

    actor: Actor
    critic: ContinuousFeatureCritic
    critic_target: ContinuousFeatureCritic
    critic_noise: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        feature_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        log_std_init: float = 0.0,
        ddpm_lr: Optional[float] = None,
        ddpm_kwargs: Optional[dict[str, Any]] = None,
        post_linear_modules: Optional[list] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        self.feature_space = feature_space
        self.ddpm_lr = ddpm_lr
        self.log_std_init = log_std_init

        if net_arch is None:
            net_arch = [512, 512]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.actor_kwargs["log_std_init"] = log_std_init

        # Action critic: operates on DDPM features phi(s, a), keeps the low-rank
        # representation learning objective alive.
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": False,
                "observation_space": self.feature_space,
                "post_linear_modules": post_linear_modules,
            }
        )

        # Noise critic: operates directly on (obs, noise), no representation involved.
        self.critic_noise_kwargs = self.net_args.copy()
        self.critic_noise_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": False,
                "post_linear_modules": post_linear_modules,
            }
        )

        if ddpm_kwargs is None:
            ddpm_kwargs = {}
        self.ddpm_kwargs = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            **ddpm_kwargs,
        }

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # create ddpm: learns the low-rank MDP transition kernel + reward
        self.ddpm = self.make_ddpm()
        self.ddpm_target = self.make_ddpm()
        self.ddpm_target.load_state_dict(self.ddpm.state_dict())
        ddpm_lr = self.ddpm_lr if self.ddpm_lr is not None else lr_schedule(1)
        self.ddpm.optimizer = self.optimizer_class(
            self.ddpm.parameters(),
            lr=ddpm_lr,
            **self.optimizer_kwargs,
        )

        # create SAC-style stochastic actor (no target network — SAC does not use actor target)
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # create action critic (DDPM feature based, with target network)
        self.critic = self.make_critic(features_extractor=None)
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # create noise critic (raw obs/noise based, no target network — it is
        # trained purely by distillation against the action critic)
        self.critic_noise = self.make_critic_noise(features_extractor=None)
        self.critic_noise.optimizer = self.optimizer_class(
            self.critic_noise.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # target networks should always be in eval mode
        self.ddpm_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)
        self.actor_target.set_training_mode(False)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousFeatureCritic:
        if features_extractor is None:
            features_extractor = FlattenExtractor(self.feature_space)
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousFeatureCritic(**critic_kwargs).to(self.device)

    def make_critic_noise(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        if features_extractor is None:
            features_extractor = FlattenExtractor(self.observation_space)
        critic_noise_kwargs = self._update_features_extractor(self.critic_noise_kwargs, features_extractor)
        return ContinuousCritic(**critic_noise_kwargs).to(self.device)

    def make_ddpm(self) -> DDPM:
        return DDPM(**self.ddpm_kwargs).to(self.device)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                log_std_init=self.log_std_init,
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self.actor._predict(observation, deterministic=deterministic)

    def forward(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        """
        self.ddpm.set_training_mode(mode)
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.critic_noise.set_training_mode(mode)
        self.training = mode


MlpPolicy = LARBRIDSeparateCriticPolicy


class CnnPolicy(LARBRIDSeparateCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        feature_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        log_std_init: float = 0.0,
        ddpm_lr: Optional[float] = None,
        ddpm_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            feature_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            log_std_init,
            ddpm_lr,
            ddpm_kwargs,
        )


class MultiInputPolicy(LARBRIDSeparateCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        feature_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        log_std_init: float = 0.0,
        ddpm_lr: Optional[float] = None,
        ddpm_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            feature_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            log_std_init,
            ddpm_lr,
            ddpm_kwargs,
        )
