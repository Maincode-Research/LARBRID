<div align="center">

# LARBRID Online Fine-Tuning

</div>

## Overview
This directory contains the online implementations of LARBRID (`larbrid`), its $\psi$-only simplification variant LARBRID-Psi (`larbrid_psi`) and the separate critic ablation (`larbrid_separate_critic`). Experiments support both Robomimic and Gymnasium tasks. Refer to the main README for more details.

## Installation
See the main README for details on installation and setup.

We use the same pre-trained diffusion policy checkpoints as DSRL and DPPO, which can be found [here](https://drive.google.com/drive/folders/1kzC49RRFOE7aTnJh_7OvJ1K5XaDmtuh1?usp=share_link). Download the contents of this folder and place them in `./dppo/log`.

## Running LARBRID
To run LARBRID on Robomimic, call
```bash
python train.py --config-path=cfg/robomimic --config-name=larbrid_can.yaml
```
where `larbrid_can.yaml` can be replaced by the config file of the desired task. Similarly, for Gym, call
```bash
python train.py --config-path=cfg/gym --config-name=larbrid_hopper.yaml
```
where `larbrid_hopper.yaml` can be replaced by the config file of the desired task.

The variants and the DSRL baseline follow the same pattern by swapping in the corresponding config, e.g. `larbrid_psi_transport.yaml`, `larbrid_separate_critic_hopper.yaml` or `dsrl_can.yaml`. All configs live in `cfg/robomimic/` and `cfg/gym/`. Hyperparameters can be overridden from the command line, e.g. `train.utd=20 seed=42`.

## Acknowledgements
Our implementation of LARBRID and its variants is built on top of [DSRL](https://github.com/ajwagen/dsrl), [DPPO](https://github.com/irom-princeton/dppo) and [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3). For our diffusion policy implementation, we use the one given in the DPPO codebase. We use the same training scripts as DSRL and DPPO to ensure fair and consistent training.

## Citation
```
@article{liang2026larbrid,
  author    = {Liang, Yuanchu and Yang, Yue and He, An-Chi and Kurniawati, Hanna},
  title     = {Efficient Diffusion Policy Fine Tuning with Latent Noise Representation Bridging},
  journal   = {Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
```
