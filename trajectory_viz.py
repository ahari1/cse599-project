"""trajectory_viz.py - visualize end-effector motion planned by DataCollector

This utility spins up a PyBullet instance (DIRECT mode) using the same
configuration as :mod:`data_collection`.  It then generates a motion
trajectory with :meth:`DataCollector._plan_motion` and records the
corresponding world-frame hand positions while walking the trajectory.

Usage::

    python trajectory_viz.py --variability medium --steps 200

The resulting figure will pop up (requires an X display); if running
headless you can add ``--out path.png`` to save the plot instead.
"""

import argparse
import numpy as np
import matplotlib
# use interactive backend for display, but allow saving to file
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import tempfile
import os
from data_collection import DataCollector


def collect_path(collector, steps, render_frames=False):
    # reset scene and compute path
    collector._reset_arm()
    collector._reset_block()
    collector._settle(30)
    block_pos = collector._get_block_position()
    start_pos, _ = collector._get_hand_state()
    traj = collector._plan_motion(start_pos, block_pos, steps)

    positions = []
    frames = []
    for jt in traj:
        collector._set_joint_targets(jt)
        collector._settle(10)  # Increased for smoother movement
        pos, _ = collector._get_hand_state()
        positions.append(pos)
        if render_frames:
            # take a snapshot from pybullet
            img = collector._get_camera_image()
            frames.append(img)
    return np.array(positions), block_pos, frames


def plot_positions(positions, block_pos, variability, out_path=None):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "-o",
            label="EE path", markersize=2)
    ax.scatter([block_pos[0]], [block_pos[1]], [block_pos[2] + 0.05],
               color="r", s=50, label="block top")
    ax.set_title(f"End-effector trajectory ({variability})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved trajectory figure to {out_path}")
    else:
        # Save to temp file and open it
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = tmp.name
        fig.savefig(temp_path, dpi=150, bbox_inches="tight")
        print(f"saved trajectory figure to {temp_path}")
        os.startfile(temp_path)  # Open the image file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variability", type=str, default="high",
                        choices=["low", "medium", "high"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", type=str, default=None,
                        help="output image path (png)")
    parser.add_argument("--render", action="store_true",
                        help="show the pybullet GUI while generating the trajectory")
    parser.add_argument("--video", type=str, default=None,
                        help="save an mp4 animation of the arm moving")
    args = parser.parse_args()

    collector = DataCollector(args.variability, 1, args.steps, output_file="/tmp/ignore.csv")
    if args.render:
        # switch to GUI mode if requested
        p.disconnect()
        collector.physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 240.0)
        collector.planeId = p.loadURDF("plane.urdf")
        collector.kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        collector.objectId = collector._spawn_block()

    positions, block_pos, frames = collect_path(collector, args.steps, render_frames=bool(args.video or args.render))
    plot_positions(positions, block_pos, args.variability, out_path=args.out)

    if args.video or args.render:
        # if video requested, write mp4
        import imageio
        if args.video:
            imageio.mimsave(args.video, frames, fps=30)
            print(f"saved video to {args.video}")
            os.startfile(args.video)  # Open the video file
        if args.render:
            # keep GUI open until user closes
            print("Press Ctrl+C or close window to exit")
            try:
                while True:
                    p.stepSimulation()
            except KeyboardInterrupt:
                pass
            p.disconnect()


if __name__ == "__main__":
    main()

