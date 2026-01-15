# Point Policy: Unifying Observations and Actions with Key Points for Robot Manipulation

This is a repository containing the code for the paper [Point Policy: Unifying Observations and Actions with Key Points for Robot Manipulation](https://arxiv.org/abs/2502.20391).

![Image](https://github.com/user-attachments/assets/a03066bc-16fc-4ce5-b1eb-e3c8dc329a2d)

## Instructions

We have provided the instructions for [installation and data collection](instructions/installation_and_data_collection.md) and [code execution](instructions/code.md) at the respective links.

## Dataset

We have made the data used for Point Policy publicly available [here](https://huggingface.co/datasets/siddhanthaldar/Point-Policy/tree/main).
This repo can also include example demo files under `data/franka_env` as `.pkl` files. The task name you pass to training/eval is the filename without the `.pkl` suffix (for example, `data/franka_env/bottle_upright.pkl` maps to `suite/task/franka_env=bottle_upright`). When using these local samples, set `data_dir` to the `data` directory (the dataloader appends `/franka_env`).

Example training command (local sample data):
```
python train.py agent=point_policy suite=point_policy dataloader=point_policy eval=false \
  suite.use_robot_points=true suite.use_object_points=true \
  suite/task/franka_env=bottle_upright experiment=point_policy
```

Example evaluation command (use a saved snapshot):
```
python eval_point_track.py agent=point_policy suite=point_policy dataloader=point_policy eval=true \
  suite.use_robot_points=true suite.use_object_points=true \
  suite/task/franka_env=bottle_upright \
  root_dir=/mnt/data/workspace/Point-Policy data_dir=/mnt/data/workspace/Point-Policy/data \
  bc_weight=/mnt/data/workspace/Point-Policy/point_policy/exp_local/2026.01.12/point_policy/deterministic/163126_hidden_dim_256/snapshot/10000.pt
```

Point labeling notes:
- Label exactly `num_object_points` points (see `point_policy/cfgs/suite/task/franka_env/<task>.yaml`).
- Keep point order consistent between `pixels1` and `pixels2` (point i in both views must match).
- Pick stable, visible object features (edges/corners), avoid background and occluded spots.

## Bibtex

If you find this work useful, please cite the paper using the following bibtex:

```
@article{haldar2025point,
  title={Point Policy: Unifying Observations and Actions with Key Points for Robot Manipulation},
  author={Haldar, Siddhant and Pinto, Lerrel},
  journal={arXiv preprint arXiv:2502.20391},
  year={2025}
}
```

## Queries/Comments/Discussions

We welcome any queries, comments or discussions on the paper. Please feel free to open an issue on this repository or reach out to siddhanthaldar@nyu.edu.
