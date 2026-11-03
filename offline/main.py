import os
import platform

import json
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('online_utd', 1, 'online gradient steps per env step.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='LARBRID-OGBench', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack, render=FLAGS.video_episodes > 0
    )
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
    if FLAGS.online_steps > 0:
        assert 'visual' not in FLAGS.env_name, 'Online fine-tuning is currently not supported for visual environments.'

    print("Loaded environment and dataset")

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    if FLAGS.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        # Use the training dataset as the replay buffer.
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] in ('rebrac',):
                dataset.return_next_actions = True

    print("Set up dataset")

    # Create agent.
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    bcf_pretrain_agent = None  # Holds BCFAgent during offline pretraining for larbrid_o2o

    if config['agent_name'] in ('larbrid', 'larbrid_o2o'):
        prior_type = config.get('prior_policy_type', 'bcf') if config['agent_name'] == 'larbrid_o2o' else 'bcf'

        if prior_type == 'fql':
            # Load distilled one-step flow from a pre-trained FQL checkpoint.
            from agents.fql import FQLAgent as _FQLAgent, get_config as _fql_get_config
            fql_dir = os.path.join('exp', 'fql', FLAGS.env_name)
            fql_config = _fql_get_config()
            # Mirror architecture params from larbrid_o2o config so the restored
            # network shape matches whatever was used to train FQL.
            for key in ('actor_hidden_dims', 'actor_layer_norm', 'encoder', 'flow_steps'):
                if key in config:
                    fql_config[key] = config[key]
            fql_agent = _FQLAgent.create(
                FLAGS.seed,
                example_batch['observations'],
                example_batch['actions'],
                fql_config,
            )
            fql_agent = restore_agent(fql_agent, fql_dir, FLAGS.online_steps)
            agent = agent_class.create(
                FLAGS.seed,
                example_batch['observations'],
                example_batch['actions'],
                config,
                bc_flow=fql_agent.network,
            )
        elif config['agent_name'] == 'larbrid_o2o' and FLAGS.offline_steps > 0:
            # Train BCF from scratch during offline phase; larbrid_o2o is created after.
            from agents.bcf import BCFAgent as _BCFAgent
            bcf_pretrain_agent = _BCFAgent.create(
                FLAGS.seed,
                example_batch['observations'],
                example_batch['actions'],
                config,
            )
            agent = None  # Placeholder; larbrid_o2o is created after offline BCF pretraining.
        else:
            # Load frozen bc_flow from the BCF checkpoint for this environment.
            from agents.bcf import BCFAgent as _BCFAgent
            bcf_dir = os.path.join('exp', 'bcf', FLAGS.env_name)
            bcf_agent = _BCFAgent.create(
                FLAGS.seed,
                example_batch['observations'],
                example_batch['actions'],
                config,
            )
            checkpoint = FLAGS.offline_steps if config['agent_name'] == 'larbrid' else FLAGS.online_steps
            print(f"checkpoint {checkpoint}")
            bcf_agent = restore_agent(bcf_agent, bcf_dir, checkpoint)
            agent = agent_class.create(
                FLAGS.seed,
                example_batch['observations'],
                example_batch['actions'],
                config,
                bc_flow=bcf_agent.bc_flow,
            )
    else:
        agent = agent_class.create(
            FLAGS.seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
        )

    # Restore agent.
    if FLAGS.restore_path is not None and agent is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    print("Intialised agent, start training...")

    # AE warm-up for larbrid_i_img: pre-train the visual encoder/decoder 
    if config['agent_name'] == 'larbrid_i_img':
        ae_warmup_steps = int(config.get('ae_warmup_steps', 10000))
        ae_warmup_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'ae_warmup.csv'))
        warmup_start = time.time()
        for w in tqdm.tqdm(range(1, ae_warmup_steps + 1), desc='AE warmup',
                           smoothing=0.1, dynamic_ncols=True):
            batch = train_dataset.sample(config['batch_size'])
            agent, warmup_info = agent.pretrain_ae(batch)
            if w % FLAGS.log_interval == 0 or w == ae_warmup_steps:
                warmup_metrics = {f'ae_warmup/{k}': v for k, v in warmup_info.items()}
                warmup_metrics['ae_warmup/time'] = time.time() - warmup_start
                if w == ae_warmup_steps:
                    recon_batch = train_dataset.sample(10)
                    input_imgs, recon_imgs = agent.reconstruct(recon_batch['observations'])
                    input_imgs = np.array(input_imgs)
                    recon_imgs = np.array(recon_imgs)
                    warmup_metrics['ae_warmup/input_images'] = [wandb.Image(input_imgs[i], caption=f'sample_{i}') for i in range(10)]
                    warmup_metrics['ae_warmup/recon_images'] = [wandb.Image(recon_imgs[i], caption=f'sample_{i}') for i in range(10)]
                wandb.log(warmup_metrics, step=w)
                ae_warmup_logger.log(warmup_metrics, step=w)
        ae_warmup_logger.close()

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            # Offline RL.
            batch = train_dataset.sample(config['batch_size'])

            if bcf_pretrain_agent is not None:
                # BCF pretraining for larbrid_o2o
                bcf_pretrain_agent, update_info = bcf_pretrain_agent.update(batch)
            elif config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)

        else:
            # Online fine-tuning
            if bcf_pretrain_agent is not None:
                # On the first online step after BCF pretraining, create larbrid_o2o.
                print("BCF pretraining complete. Creating LARBRIDO2OAgent for online fine-tuning...")
                agent = agent_class.create(
                    FLAGS.seed,
                    example_batch['observations'],
                    example_batch['actions'],
                    config,
                    bc_flow=bcf_pretrain_agent.bc_flow,
                )
                bcf_pretrain_agent = None

            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()

            action = agent.sample_actions(observations=ob, temperature=1, seed=key)
            action = np.array(action)

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                # Adjust reward for D4RL antmaze.
                reward = reward - 1.0

            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Update agent.
            update_count = 0
            while update_count < FLAGS.online_utd:
                if FLAGS.balanced_sampling:
                    # Half-and-half sampling from the training dataset and the replay buffer.
                    dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                    replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                    batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
                else:
                    batch = replay_buffer.sample(config['batch_size'])

                if config['agent_name'] == 'rebrac':
                    agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
                else:
                    agent, update_info = agent.update(batch)

                update_count += 1

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                val_agent = bcf_pretrain_agent if bcf_pretrain_agent is not None else agent
                _, val_info = val_agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        eval_agent = bcf_pretrain_agent if bcf_pretrain_agent is not None else agent
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0) and eval_agent is not None:
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=eval_agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            #save_agent(agent, FLAGS.save_dir, i)
            if config['agent_name'] == 'bcf':
                bcf_save_dir = os.path.join('exp', 'bcf', FLAGS.env_name)
                os.makedirs(bcf_save_dir, exist_ok=True)
                save_agent(agent, bcf_save_dir, i)
            elif config['agent_name'] == 'fql':
                fql_save_dir = os.path.join('exp', 'fql', FLAGS.env_name)
                os.makedirs(fql_save_dir, exist_ok=True)
                save_agent(agent, fql_save_dir, i)
            elif config['agent_name'] == 'larbrid_i':
                larbrid_i_save_dir = os.path.join('exp', 'larbrid_i', FLAGS.env_name)
                os.makedirs(larbrid_i_save_dir, exist_ok=True)
                save_agent(agent, larbrid_i_save_dir, i)
            elif config['agent_name'] == 'larbrid_o2o' and agent is not None:
                larbrid_o2o_save_dir = os.path.join('exp', 'larbrid_o2o', FLAGS.env_name)
                os.makedirs(larbrid_o2o_save_dir, exist_ok=True)
                save_agent(agent, larbrid_o2o_save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
