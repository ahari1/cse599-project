"""trajectory_viz.py - visualize end-effector motion planned by DataCollector

Spins up a PyBullet DIRECT session using the same DataCollector used for data
collection, runs one episode (respecting obstacle placement, waypoints, etc.),
and plots the 3-D end-effector path.

Usage::

    conda run --no-capture-output -p ./conda_env python trajectory_viz.py --variability high --steps 200

Options
-------
--variability   low | medium | high   (default: high)
--steps         int   total sim steps to record  (default: 200)
--out           str   save figure to this path instead of displaying
"""

import argparse
import os
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; we save to file then open
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import pybullet as p
from data_collection import DataCollector


# ── helpers ───────────────────────────────────────────────────────────────────

def _open(path):
    """Cross-platform file open (macOS / Linux / Win)."""
    if os.name == "nt":
        os.startfile(path)
    elif os.uname().sysname == "Darwin":
        subprocess.call(["open", path])
    else:
        subprocess.call(["xdg-open", path])


def collect_episode(collector: DataCollector, steps: int):
    """Run one episode and return the recorded EE positions + scene info.

    Returns
    -------
    positions   : np.ndarray  (N, 3)  world-frame hand XYZ at each sim step
    block_pos   : list | None         block centre (None for free-motion)
    obs_pos     : list | None         obstacle centre (None when absent)
    waypoints   : list[list]          the IK target positions used
    """
    import random as _random

    # ── reset ──────────────────────────────────────────────────────────────
    collector._reset_arm()
    collector._reset_block()
    collector._settle(30)

    cfg = collector.cfg

    # ── decide episode type ────────────────────────────────────────────────
    is_free_motion = _random.random() < cfg["no_object_prob"]
    if is_free_motion:
        if collector.objectId is not None:
            p.removeBody(collector.objectId)
            collector.objectId = None
        block_pos = None
    else:
        block_pos = collector._get_block_position()

    # ── maybe place obstacle ───────────────────────────────────────────────
    if collector.obstacleId is not None:
        p.removeBody(collector.obstacleId)
        collector.obstacleId = None
    obs_pos = None
    if not is_free_motion and _random.random() < cfg["obstacle_prob"]:
        collector.obstacleId = collector._spawn_obstacle(block_pos)
        collector._settle(10)
        obs_pos, _ = p.getBasePositionAndOrientation(collector.obstacleId)
        obs_pos = list(obs_pos)

    # ── plan waypoints ─────────────────────────────────────────────────────
    if is_free_motion:
        waypoints = [collector._sample_free_target()]
    else:
        waypoints = collector._plan_waypoints(block_pos)

    # ── simulate and record ────────────────────────────────────────────────
    n_wps = len(waypoints)
    base_steps = steps // n_wps
    leftover   = steps - base_steps * n_wps

    positions = []
    for wp_idx, wp_pos in enumerate(waypoints):
        jt = collector._ik_joints(wp_pos)
        wp_steps = base_steps + (leftover if wp_idx == n_wps - 1 else 0)
        for _ in range(wp_steps):
            collector._set_joint_targets(jt)
            p.stepSimulation()
            pos, _ = collector._get_hand_state()
            positions.append(pos)

    return np.array(positions), block_pos, obs_pos, waypoints


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_trajectory(positions, block_pos, obs_pos, waypoints, variability, out_path):
    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # EE path
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            "-o", markersize=2, linewidth=1.2, label="EE path", zorder=3)

    # Waypoint targets
    wps = np.array(waypoints)
    ax.scatter(wps[:, 0], wps[:, 1], wps[:, 2],
               marker="^", s=80, color="green", zorder=5,
               label=f"waypoints ({len(waypoints)})")
    for i, (x, y, z) in enumerate(wps):
        ax.text(x, y, z + 0.01, f"wp{i}", fontsize=7, color="green")

    # Block
    if block_pos is not None:
        ax.scatter([block_pos[0]], [block_pos[1]], [block_pos[2] + 0.05],
                   color="red", s=120, marker="s", zorder=5, label="block top")

    # Obstacle
    if obs_pos is not None:
        ax.scatter([obs_pos[0]], [obs_pos[1]], [obs_pos[2]],
                   color="orange", s=200, marker="o", zorder=5,
                   label="obstacle", alpha=0.7)

    # Origin (robot base)
    ax.scatter([0], [0], [0], color="black", s=80, marker="x",
               zorder=5, label="robot base")

    ax.set_title(f"End-effector trajectory  [{variability}]")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved → {out_path}")
    _open(out_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variability", default="high",
                        choices=["low", "medium", "high"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out",   type=str, default=None,
                        help="Save figure here (png).  Defaults to a temp file.")
    args = parser.parse_args()

    print(f"[trajectory_viz] variability={args.variability}  steps={args.steps}")
    collector = DataCollector(args.variability, 1, args.steps,
                              output_file="/tmp/_traj_ignore.csv")

    positions, block_pos, obs_pos, waypoints = collect_episode(collector, args.steps)

    ep_type = ("free-motion" if block_pos is None
               else ("contact+obstacle" if obs_pos else "contact"))
    print(f"  episode_type : {ep_type}")
    print(f"  waypoints    : {len(waypoints)}")
    if block_pos:
        print(f"  block_pos    : {[round(v,3) for v in block_pos]}")
    if obs_pos:
        print(f"  obstacle_pos : {[round(v,3) for v in obs_pos]}")
    print(f"  EE positions : {len(positions)} steps recorded")

    out = args.out
    if out is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False,
                                         prefix="traj_viz_")
        out = tmp.name
        tmp.close()

    plot_trajectory(positions, block_pos, obs_pos, waypoints,
                    args.variability, out)


if __name__ == "__main__":
    main()
