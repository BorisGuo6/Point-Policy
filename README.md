# Point Policy: Unifying Observations and Actions with Key Points for Robot Manipulation

This is a repository containing the code for the paper [Point Policy: Unifying Observations and Actions with Key Points for Robot Manipulation](https://arxiv.org/abs/2502.20391).

![Image](https://github.com/user-attachments/assets/a03066bc-16fc-4ce5-b1eb-e3c8dc329a2d)

## Instructions

We have provided the instructions for [installation and data collection](instructions/installation_and_data_collection.md) and [code execution](instructions/code.md) at the respective links.

## Quick Start (Local Sample Data)

1) Set paths in configs:
- `point_policy/cfgs/config.yaml`: `root_dir`, `data_dir`
- `point_policy/cfgs/config_eval.yaml`: `root_dir`, `data_dir` (if you override the defaults)
- `point_policy/cfgs/suite/points_cfg.yaml`: `root_dir`

2) Local sample tasks live in `data/franka_env/*.pkl`. Example task names:
`bottle_on_rack`, `bottle_upright`, `bowl_in_oven`, `bread_on_plate`, `close_oven`, `drawer_close`, `fold_towel`, `sweep_broom`.

3) Training (local sample data):
```
python train.py agent=point_policy suite=point_policy dataloader=point_policy eval=false \
  suite.use_robot_points=true suite.use_object_points=true \
  suite/task/franka_env=bottle_upright experiment=point_policy
```

4) Offline evaluation (no robot required):
```
python offline_eval.py agent=point_policy suite=point_policy dataloader=point_policy \
  suite.use_robot_points=true suite.use_object_points=true \
  suite/task/franka_env=bottle_upright \
  root_dir=/mnt/data/workspace/Point-Policy data_dir=/mnt/data/workspace/Point-Policy/data \
  bc_weight=/mnt/data/workspace/Point-Policy/point_policy/exp_local/2026.01.12/point_policy/deterministic/163126_hidden_dim_256/snapshot/10000.pt
```

Outputs are written under `point_policy/exp_local/...` with snapshots in `snapshot/`.

## Robot Evaluation (Optional)

`eval_point_track.py` uses the real robot if `eval=true`. Make sure Franka servers are running and point coordinates are prepared.

Example evaluation command (real robot):
```
python eval_point_track.py agent=point_policy suite=point_policy dataloader=point_policy eval=true \
  suite.use_robot_points=true suite.use_object_points=true \
  suite/task/franka_env=bottle_upright \
  root_dir=/mnt/data/workspace/Point-Policy data_dir=/mnt/data/workspace/Point-Policy/data \
  bc_weight=/mnt/data/workspace/Point-Policy/point_policy/exp_local/2026.01.12/point_policy/deterministic/163126_hidden_dim_256/snapshot/10000.pt
```

## Point Labeling (Object Points)

Use `point_policy/robot_utils/franka/label_points.ipynb` to generate point coordinates:

1) Set `task_name`, `object_name`, `pickle_path`, and `traj_idx` in the first code cell.
2) Run the cell with `pixel_key = "pixels1"`, click points, then press "Save Points".
3) Repeat for `pixel_key = "pixels2"` using the same point order.

Files are saved to:
- `coordinates/<task>/images/pixels1.png`
- `coordinates/<task>/coords/pixels1_<object_name>.pkl`
- `coordinates/<task>/images/pixels2.png`
- `coordinates/<task>/coords/pixels2_<object_name>.pkl`

Point labeling notes:
- Label exactly `num_object_points` points (see `point_policy/cfgs/suite/task/franka_env/<task>.yaml`).
- Keep point order consistent between `pixels1` and `pixels2` (point i in both views must match).
- Pick stable, visible object features (edges/corners), avoid background and occluded spots.

## DIFT Model

Object-point tracking uses DIFT from the `dift` submodule. The default SD model is `Manojb/stable-diffusion-2-1-base` (Diffusers format).
The model will download on first use; you may need to log in to Hugging Face:
```
python -m huggingface_hub.cli.hf auth login
```

## Dataset

We have made the data used for Point Policy publicly available [here](https://huggingface.co/datasets/siddhanthaldar/Point-Policy/tree/main).

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
