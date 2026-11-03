# load checkpoints and generate videos in the evaluation mode
import os
import random

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'videos/', 'Directory to save rendered videos.')
flags.DEFINE_string('restore_path', None, 'Restore path (required).')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch (required).')
flags.DEFINE_integer('bcf_restore_epoch', None, 'BCF checkpoint epoch (required for larbrid_o2o and other BCF-prior agents).')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps (used only for checkpoint path resolution).')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps (used only for checkpoint path resolution).')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 0, 'Number of evaluation episodes (non-rendered).')
flags.DEFINE_integer('video_episodes', 5, 'Number of video episodes to render.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('render_width', 480, 'Width of the rendered video frames.')
flags.DEFINE_integer('render_height', 480, 'Height of the rendered video frames.')
flags.DEFINE_integer('online_utd', 1, 'Online gradient steps per env step.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


def save_renders(renders, save_dir, fps=15):
    """Save a list of rendered episode frames as MP4 files using moviepy."""
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    os.makedirs(save_dir, exist_ok=True)
    for i, frames in enumerate(renders):
        path = os.path.join(save_dir, f'episode_{i:03d}.mp4')
        clip = ImageSequenceClip([frame for frame in frames], fps=fps)
        clip.write_videofile(path, logger=None)
        print(f'Saved video: {path}')


def main(_):
    assert FLAGS.restore_path is not None, 'restore_path must be specified.'
    assert FLAGS.restore_epoch is not None, 'restore_epoch must be specified.'
    assert FLAGS.video_episodes > 0, 'video_episodes must be > 0.'
    assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench singletask environments.'

    config = FLAGS.agent

    # Make environment with render mode enabled.
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name,
        frame_stack=FLAGS.frame_stack,
        render=True,
        render_width=FLAGS.render_width,
        render_height=FLAGS.render_height,
    )

    # Set up random seeds.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Get an example batch for agent initialisation (shape inference only).
    train_dataset = Dataset.create(**train_dataset)
    example_batch = train_dataset.sample(1)

    # Create agent (mirrors the same logic as main.py).
    agent_class = agents[config['agent_name']]

    if config['agent_name'] in ('larbrid', 'larbrid_o2o'):
        prior_type = config.get('prior_policy_type', 'bcf') if config['agent_name'] == 'larbrid_o2o' else 'bcf'

        if prior_type == 'fql':
            from agents.fql import FQLAgent as _FQLAgent, get_config as _fql_get_config
            fql_dir = os.path.join('exp', 'fql', FLAGS.env_name)
            fql_config = _fql_get_config()
            for key in ('actor_hidden_dims', 'actor_layer_norm', 'encoder', 'flow_steps'):
                if key in config:
                    fql_config[key] = config[key]
            fql_agent = _FQLAgent.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], fql_config)
            fql_agent = restore_agent(fql_agent, fql_dir, FLAGS.online_steps)
            agent = agent_class.create(
                FLAGS.seed, example_batch['observations'], example_batch['actions'], config, bc_flow=fql_agent.network
            )
        else:
            assert FLAGS.bcf_restore_epoch is not None, (
                f'--bcf_restore_epoch is required for {config["agent_name"]} with BCF prior.'
            )
            from agents.bcf import BCFAgent as _BCFAgent
            bcf_dir = os.path.join('exp', 'bcf', FLAGS.env_name)
            bcf_agent = _BCFAgent.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], config)
            bcf_agent = restore_agent(bcf_agent, bcf_dir, FLAGS.bcf_restore_epoch)
            agent = agent_class.create(
                FLAGS.seed, example_batch['observations'], example_batch['actions'], config,
                bc_flow=bcf_agent.bc_flow,
            )
    else:
        agent = agent_class.create(
            FLAGS.seed, example_batch['observations'], example_batch['actions'], config
        )

    # Restore the target checkpoint.
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    print(f'Restored agent from {FLAGS.restore_path} epoch {FLAGS.restore_epoch}.')

    # Run evaluation with rendering.
    print(f'Rendering {FLAGS.video_episodes} episode(s)...')
    _, _, renders = evaluate(
        agent=agent,
        env=eval_env,
        config=config,
        num_eval_episodes=FLAGS.eval_episodes,
        num_video_episodes=FLAGS.video_episodes,
        video_frame_skip=FLAGS.video_frame_skip,
    )

    save_renders(renders, FLAGS.save_dir)
    print(f'Done. {len(renders)} video(s) saved to {FLAGS.save_dir}')


if __name__ == '__main__':
    app.run(main)
