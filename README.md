# Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals.

This repository contains the code for the paper [Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals](https://arxiv.org/abs/2601.19810) (ICLR 2026). It includes code for the main method (**ULEE**) and baselines in the paper, as well as for all the experiments. ULEE is an unsupervised pre-training method in which an intrinsically motivated agent meta-learns an adaptive policy from a curriculum
of self-imposed goals. See full paper for details.

The repository includes JAX implementations of the following baselines: Diversity is All You Need ([DIAYN](https://arxiv.org/abs/1802.06070)), Exploration by Random Network Distillation ([RND](https://arxiv.org/abs/1810.12894)),  Reinforcement Learning Squared ([RL²](https://arxiv.org/abs/1611.02779)), and vanilla Proximal Policy Optimization ([PPO](https://arxiv.org/abs/1707.06347)) with no pre-training. Ablations for ULEE can be reproduced by adjusting its training configurations.

## Code Structure

Each method in the paper has its own directory (`ULEE`, `DIAYN`, `RND`, `RL2`, `PPO`) with the code needed for training. These have a closely related internal structure with at least the following files:
* `config.py`
* `setups.py`
* `data_collection_and_updates.py`
* `main_loop.py`

The methods additionally rely on code from `networks` and `shared_code` directories.

The `evaluations` directory contains the code for performing the different evaluations on each method.

The `experiments` directory contains notebooks for 1) running the training of each method, and 2) running evaluations on each method. These should facilitate reproducing the paper's results.

 -  For ULEE, both standard PPO and DIAYN (with access to adversarial difficulty-seeking rewards) can be used for the Goal-search Policy. ULEE's Difficulty Predictor (from the paper's nomenclature) is sometimes referred to as the 'judge' network in the codebase.
 - For `RND`, the code for the network architectures and for executing training runs is within the RND folder itself.


 ## Acknowledgments

 The code for the policies' base Transformer-XL architecture, and the code in `ppo_update.py` is inspired from [transformerXL_PPO_JAX](https://github.com/Reytuag/transformerXL_PPO_JAX/tree/main), which itself acknowledges the following sources of inspiration:
- [PureJaxRL](https://github.com/luchris429/purejaxrl)
- [Huggingface transformerXL](https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/deprecated/transfo_xl/modeling_transfo_xl.py)
- https://github.com/MarcoMeter/episodic-transformer-memory-ppo

Experiments are performed on [XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid) environments. Our code takes inspiration from their repository on how to best interact with environments and modify their behavior (e.g., environment wrappers).