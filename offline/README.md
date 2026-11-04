<div align="center">

# LARBRID Offline Fine-Tuning

</div>

## Overview

This directory contains the offline implementations of LARBRID-I (`larbrid_i`), LARBRID (`larbrid`) and the offline-to-online variant LARBRID-O2O (`larbrid_o2o`), together with the baselines they are compared against. Refer to the main README for more details.

## Installation

See the [main README](https://github.com/Maincode-Research/LARBRID/blob/main/README.md) for details on installation and setup.

## Usage

To run LARBRID, first train the behaviour cloning flow policy:
```bash
python main.py --agent=agents/bcf.py --env_name=antmaze-large-navigate-singletask-v0
```

Then run LARBRID, which fine-tunes the frozen flow model:
```bash
python main.py --agent=agents/larbrid.py --env_name=antmaze-large-navigate-singletask-v0 --agent.discount=0.99
```

To run LARBRID-I,
```bash
python main.py --agent=agents/larbrid_i.py --env_name=antmaze-large-navigate-singletask-v0 --agent.discount=0.99 --agent.alpha=10

python main.py --agent=agents/larbrid_i.py --env_name=antsoccer-arena-navigate-singletask-v0 --agent.discount=0.995 --agent.alpha=10

python main.py --agent=agents/larbrid_i.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent.discount=0.995 --agent.alpha=30

python main.py --agent=agents/larbrid_i.py --env_name=cube-single-play-singletask-v0 --agent.discount=0.99 --agent.alpha=300
```

To run LARBRID-O2O, first train the behaviour cloning flow policy as above, since it is loaded as the frozen prior from `exp/bcf/<env_name>/`. Then run,
```bash
python main.py --agent=agents/larbrid_o2o.py --env_name=antmaze-large-navigate-singletask-v0 --offline_steps=0 --online_steps=1000000 --agent.discount=0.99 --balanced_sampling=1

python main.py --agent=agents/larbrid_o2o.py --env_name=antsoccer-arena-navigate-singletask-v0 --offline_steps=0 --online_steps=1000000 --agent.discount=0.995 --balanced_sampling=1

python main.py --agent=agents/larbrid_o2o.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --offline_steps=0 --online_steps=1000000 --agent.discount=0.995 --balanced_sampling=1

python main.py --agent=agents/larbrid_o2o.py --env_name=scene-play-singletask-v0 --offline_steps=0 --online_steps=1000000 --agent.discount=0.995 --balanced_sampling=1

python main.py --agent=agents/larbrid_o2o.py --env_name=cube-double-play-singletask-v0 --offline_steps=0 --online_steps=1000000 --agent.discount=0.99 --balanced_sampling=1
```

To run LARBRID-I on visual observations,
```bash
python main.py --agent=agents/larbrid_i_img.py --env_name=visual-cube-single-play-singletask-task1-v0 --offline_steps=500000 --agent.alpha=300 --frame_stack=3
```

To restore a saved agent and render video clips,
```bash
python video.py --env_name=puzzle-4x4-play-singletask-v0 --restore_path='exp/larbrid_o2o/puzzle-4x4-play-singletask-v0' --restore_epoch=1000000 --video_episodes=3 --save_dir=videos/larbrid_o2o/puzzle-4x4 --agent=agents/larbrid_o2o.py --bcf_restore_epoch=1000000
```

## Acknowledgements

This codebase is built on top of the reference implementations of [FQL](https://github.com/seohongpark/fql/tree/master) and [OGBench](https://github.com/seohongpark/ogbench).
