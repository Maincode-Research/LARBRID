### LARBRID
# -------------------------------------------------
# we use low rank MDP formulation to extract RL
# features, by only matching this latent feature
# we can steer our diffusion policy efficiently
# as the rest of components all depends on the 
# learnt feature.
# -------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore")
import math
import torch
import random
import wandb
import numpy as np
import hydra
from omegaconf import OmegaConf
import gymnasium as gym
import sys
sys.path.append('./dppo')
 
from stable_baselines3 import SAC, DSRL, LARBRID, LARBRIDSeparateCritic, LARBRIDPsi
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env_utils import DiffusionPolicyEnvWrapper, ObservationWrapperRobomimic, ObservationWrapperGym, ActionChunkWrapper, make_robomimic_env
from utils import load_base_policy, load_offline_data, collect_rollouts, LoggingCallback

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))


@hydra.main(
	config_path=os.path.join(base_path, "cfg/robomimic"), config_name="dsrl_can.yaml", version_base=None
)
def main(cfg: OmegaConf):
	OmegaConf.resolve(cfg)

	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	if cfg.use_wandb:
		wandb.init(
			project=cfg.wandb.project,
			name=cfg.name,
			group=cfg.wandb.group,
			monitor_gym=True,
			save_code=True,
			config=OmegaConf.to_container(cfg, resolve=True),
		)

	MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)

	num_env = cfg.env.n_envs
	# Map d4rl dataset names to their gymnasium equivalents (drops mujoco_py dependency) TODO: improve this later
	_D4RL_TO_GYMNASIUM = {
		'halfcheetah-medium-v2': 'HalfCheetah-v5',
		'hopper-medium-v2': 'Hopper-v5',
		'walker2d-medium-v2': 'Walker2d-v5',
	}

	def make_env():
		if cfg.env_name in _D4RL_TO_GYMNASIUM:
			env = gym.make(_D4RL_TO_GYMNASIUM[cfg.env_name])
			env = ObservationWrapperGym(env, cfg.normalization_path)
		elif cfg.env_name in ['lift', 'can', 'square', 'transport']:
			env = make_robomimic_env(env=cfg.env_name, normalization_path=cfg.normalization_path, low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path)
			env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
		env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
		return env

	base_policy = load_base_policy(cfg)
	env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
	if cfg.algorithm == 'dsrl_sac':
		env = DiffusionPolicyEnvWrapper(env, cfg, base_policy)

	# stable-baselines3 has a double seed convention. It firstly stores the seed
	# with env.seed(seed). Internally, it then calls reset(seed) once at the beginning
	# of learning to set the seed properly. This seed is cleared for later resets to comply
	# with gymnasium's new convetion.
	env.seed(cfg.seed + 1) 
	post_linear_modules = None
	if cfg.train.use_layer_norm:
		post_linear_modules = [torch.nn.LayerNorm]

	net_arch = []
	for _ in range(cfg.train.num_layers):
		net_arch.append(cfg.train.layer_size)
	policy_kwargs = dict(
		net_arch=dict(pi=net_arch, qf=net_arch),
		activation_fn=torch.nn.Tanh,
		n_critics=cfg.train.n_critics,
		log_std_init = 0.0,
		post_linear_modules = post_linear_modules
	)
	if cfg.algorithm == 'dsrl_sac':
		model = SAC(
			"MlpPolicy",
			env,
			learning_rate=cfg.train.actor_lr,
			buffer_size=20000000,      # Replay buffer size
			learning_starts=1,    # How many steps before learning starts (total steps for all env combined)
			batch_size=cfg.train.batch_size,
			tau=cfg.train.tau,                # Target network update rate
			gamma=cfg.train.discount,               # Discount factor
			train_freq=cfg.train.train_freq,             # Update the model every train_freq steps
			gradient_steps=cfg.train.utd,         # How many gradient steps to do at each update
			action_noise=None,        # No additional action noise
			optimize_memory_usage=False,
			ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,          # Automatic entropy tuning
			target_update_interval=1, # Update target network every interval
			target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,    # Automatic target entropy
			use_sde=False,
			sde_sample_freq=-1,
			tensorboard_log=cfg.logdir,
			verbose=1,
			policy_kwargs=policy_kwargs,
		)
	elif cfg.algorithm == 'dsrl_na':
		model = DSRL(
			"MlpPolicy",
			env,
			learning_rate=cfg.train.actor_lr,
			buffer_size=10000000,      # Replay buffer size
			learning_starts=1,    # How many steps before learning starts (total steps for all env combined)
			batch_size=cfg.train.batch_size,
			tau=cfg.train.tau,                # Target network update rate
			gamma=cfg.train.discount,               # Discount factor
			train_freq=cfg.train.train_freq,             # Update the model every train_freq steps
			gradient_steps=cfg.train.utd,         # How many gradient steps to do at each update
			action_noise=None,        # No additional action noise
			optimize_memory_usage=False,
			ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,          # Automatic entropy tuning
			target_update_interval=1, # Update target network every interval
			target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,    # Automatic target entropy
			use_sde=False,
			sde_sample_freq=-1,
			tensorboard_log=cfg.logdir,
			verbose=1,
			policy_kwargs=policy_kwargs,
			diffusion_policy=base_policy,
			diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
			noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
			critic_backup_combine_type=cfg.train.critic_backup_combine_type,
		)
	elif cfg.algorithm == "larbrid":
		actor_arch = [cfg.train.layer_size] * cfg.train.actor_num_layers
		critic_arch = [cfg.train.layer_size] * cfg.train.critic_num_layers
		policy_kwargs["net_arch"] = dict(pi=actor_arch, qf=critic_arch)
		ddpm_cfg = OmegaConf.to_container(cfg.train.ddpm, resolve=True)
		policy_kwargs["ddpm_kwargs"] = ddpm_cfg
		feature_space = gym.spaces.Box(-np.inf, np.inf, (cfg.train.ddpm.feature_dim, ))
		model = LARBRID(
			"MlpPolicy",
			env,
			feature_space,
			learning_rate = cfg.train.actor_lr,
			buffer_size = 10000000,
			learning_starts=1,
			batch_size=cfg.train.batch_size,
			tau=cfg.train.tau,
			gamma=cfg.train.discount,
			train_freq=cfg.train.train_freq,
			gradient_steps=cfg.train.utd,
			action_noise=None,
			optimize_memory_usage=False,
			ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
			target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
			target_update_interval=1,
			tensorboard_log=cfg.logdir,
			verbose=1,
			policy_kwargs=policy_kwargs,
			diffusion_policy=base_policy,
			diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
			critic_backup_combine_type=cfg.train.critic_backup_combine_type,
			noise_feature_grad_steps=cfg.train.noise_feature_grad_steps,
			ddpm_lr=cfg.train.ddpm_lr,
		)
	elif cfg.algorithm == "larbrid_separate_critic":
		actor_arch = [cfg.train.layer_size] * cfg.train.actor_num_layers
		critic_arch = [cfg.train.layer_size] * cfg.train.critic_num_layers
		policy_kwargs["net_arch"] = dict(pi=actor_arch, qf=critic_arch)
		ddpm_cfg = OmegaConf.to_container(cfg.train.ddpm, resolve=True)
		policy_kwargs["ddpm_kwargs"] = ddpm_cfg
		feature_space = gym.spaces.Box(-np.inf, np.inf, (cfg.train.ddpm.feature_dim, ))
		model = LARBRIDSeparateCritic(
			"MlpPolicy",
			env,
			feature_space,
			learning_rate = cfg.train.actor_lr,
			buffer_size = 10000000,
			learning_starts=1,
			batch_size=cfg.train.batch_size,
			tau=cfg.train.tau,
			gamma=cfg.train.discount,
			train_freq=cfg.train.train_freq,
			gradient_steps=cfg.train.utd,
			action_noise=None,
			optimize_memory_usage=False,
			ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
			target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
			target_update_interval=1,
			tensorboard_log=cfg.logdir,
			verbose=1,
			policy_kwargs=policy_kwargs,
			diffusion_policy=base_policy,
			diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
			critic_backup_combine_type=cfg.train.critic_backup_combine_type,
			noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
			ddpm_lr=cfg.train.ddpm_lr,
		)
	elif cfg.algorithm == "larbrid_psi":
		actor_arch = [cfg.train.layer_size] * cfg.train.actor_num_layers
		critic_arch = [cfg.train.layer_size] * cfg.train.critic_num_layers
		policy_kwargs["net_arch"] = dict(pi=actor_arch, qf=critic_arch)
		ddpm_cfg = OmegaConf.to_container(cfg.train.ddpm, resolve=True)
		policy_kwargs["ddpm_kwargs"] = ddpm_cfg
		feature_space = gym.spaces.Box(-np.inf, np.inf, (cfg.train.ddpm.feature_dim, ))
		model = LARBRIDPsi(
			"MlpPolicy",
			env,
			feature_space,
			learning_rate=cfg.train.actor_lr,
			buffer_size=10000000,
			learning_starts=1,
			batch_size=cfg.train.batch_size,
			tau=cfg.train.tau,
			gamma=cfg.train.discount,
			train_freq=cfg.train.train_freq,
			gradient_steps=cfg.train.utd,
			action_noise=None,
			optimize_memory_usage=False,
			ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
			target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
			target_update_interval=1,
			tensorboard_log=cfg.logdir,
			verbose=1,
			policy_kwargs=policy_kwargs,
			diffusion_policy=base_policy,
			diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
			critic_backup_combine_type=cfg.train.critic_backup_combine_type,
			ddpm_lr=cfg.train.ddpm_lr,
		)

	checkpoint_callback = CheckpointCallback(
		save_freq=cfg.save_model_interval, 
		save_path=cfg.logdir+'/checkpoint/',
		name_prefix='ft_policy',
		save_replay_buffer=cfg.save_replay_buffer, 
		save_vecnormalize=True,
	)

	num_env_eval = cfg.env.n_eval_envs
	eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
	if cfg.algorithm == 'dsrl_sac':
		eval_env = DiffusionPolicyEnvWrapper(eval_env, cfg, base_policy)
	eval_env.seed(cfg.seed + num_env + 1) 

	logging_callback = LoggingCallback(
		action_chunk = cfg.act_steps, 
		eval_episodes = int(cfg.num_evals / num_env_eval), 
		log_freq=MAX_STEPS, 
		use_wandb=cfg.use_wandb, 
		eval_env=eval_env, 
		eval_freq=cfg.eval_interval,
		num_train_env=num_env,
		num_eval_env=num_env_eval,
		rew_offset=cfg.env.reward_offset,
		algorithm=cfg.algorithm,
		max_steps=MAX_STEPS,
		deterministic_eval=cfg.deterministic_eval,
	)

	logging_callback.evaluate(model, deterministic=False)
	if cfg.deterministic_eval:
		logging_callback.evaluate(model, deterministic=True)

	if cfg.load_offline_data:
		load_offline_data(model, cfg.offline_data_path, num_env)
	if cfg.train.init_rollout_steps > 0:
		collect_rollouts(model, env, cfg.train.init_rollout_steps, base_policy, cfg)	
		logging_callback.set_timesteps(cfg.train.init_rollout_steps * num_env * cfg.act_steps)

	callbacks = [checkpoint_callback, logging_callback]
	# Train the agent
	model.learn(
		total_timesteps=getattr(cfg, "total_timesteps", 20000000),
		callback = callbacks
	)

	# Save the final model
	if len(cfg.name) > 0:
		model.save(cfg.logdir+"/checkpoint/final")

	# Close environment and wandb
	env.close()
	if cfg.use_wandb:
		wandb.finish()


if __name__ == "__main__":
	main()
