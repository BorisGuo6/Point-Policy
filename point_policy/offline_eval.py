#!/usr/bin/env python3

import os
import warnings
from pathlib import Path

import hydra
import torch

import utils
from replay_buffer import make_expert_replay_loader

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _build_obs_shape(cfg):
    obs_shape = {}
    height, width = cfg.suite.img_size[1], cfg.suite.img_size[0]
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = (height, width, 3)
    obs_shape[cfg.suite.feature_key] = (cfg.suite.task_make_fn.max_state_dim,)
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = (cfg.suite.task_make_fn.max_state_dim,)
    return obs_shape


def _prepare_batch(agent, data):
    past_tracks = data["past_tracks"].float()
    future_tracks = data["future_tracks"].float()
    action_masks = data["action_mask"].float()

    if agent.pred_gripper:
        past_gripper_states = data["past_gripper_states"].float()
        future_gripper_states = data["future_gripper_states"].float()
        gripper_mask = torch.ones_like(action_masks)[:, :1]
        action_masks = torch.cat([action_masks, gripper_mask], dim=1)

    # reshape for evaluation (same as training)
    shape = past_tracks.shape
    past_tracks = past_tracks.transpose(1, 2).reshape(shape[0], shape[2], -1)
    future_tracks = future_tracks[:, 0]

    if agent.pred_gripper:
        past_gripper_states = past_gripper_states[:, None]
        future_gripper_states = future_gripper_states[:, :1]

        past_gripper_states = past_gripper_states.repeat(1, 1, agent._act_dim)
        future_gripper_states = future_gripper_states.repeat(1, 1, agent._act_dim)

        past_tracks = torch.cat([past_tracks, past_gripper_states], dim=1)
        future_tracks = torch.cat([future_tracks, future_gripper_states], dim=1)

    return past_tracks, future_tracks, action_masks


@hydra.main(config_path="cfgs", config_name="config_eval")
def main(cfg):
    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    dataset_iterable = hydra.utils.call(cfg.expert_dataset)
    loader = make_expert_replay_loader(dataset_iterable, cfg.batch_size)
    dataset = loader.dataset

    cfg.suite.task_make_fn.max_episode_len = dataset._max_episode_len
    cfg.suite.task_make_fn.max_state_dim = dataset._max_state_dim

    cfg.agent.obs_shape = _build_obs_shape(cfg)
    cfg.agent.action_shape = (cfg.suite.point_dim,)
    agent = hydra.utils.instantiate(cfg.agent)

    # Load weights
    snapshot = Path(cfg.bc_weight)
    if not snapshot.exists():
        raise FileNotFoundError(f"bc weight not found: {snapshot}")
    payload = torch.load(snapshot, map_location=device)
    agent.load_snapshot(payload, eval=True)
    agent.train(False)

    num_batches = int(getattr(cfg, "eval_batches", 100))
    losses = []
    for step, batch in enumerate(loader):
        if step >= num_batches:
            break
        data = utils.to_torch(batch, device)
        past_tracks, future_tracks, action_masks = _prepare_batch(agent, data)
        with torch.no_grad():
            past_tracks = agent.point_projector(past_tracks)
            stddev = utils.schedule(agent.stddev_schedule, step)
            _, actor_loss = agent.actor(
                past_tracks,
                stddev,
                future_tracks,
                action_masks,
            )
        losses.append(actor_loss["actor_loss"].item())

    mean_loss = sum(losses) / max(1, len(losses))
    print(f"offline_eval_batches={len(losses)} mean_actor_loss={mean_loss:.6f}")


if __name__ == "__main__":
    main()
