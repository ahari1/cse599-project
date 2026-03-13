import pybullet as p
import pybullet_data
import numpy as np
import csv
import random
import argparse


# Variability configs


# Note: I asked Gemini for the configs, so these may not be the best for testing
TIER_CONFIG = {
    "low": {
        # Block placement (x,y) ranges
        "obj_x": (0.50, 0.50),
        "obj_y": (0.00, 0.00),
        "obj_z": (0.05, 0.05),
        # Block orientation (yaw) range, radians
        "obj_yaw": (0.0, 0.0),         # fixed orientation
        # Jitter added to each joint's home angle at episode start
        "arm_jitter": 0.02,
        # Gaussian std dev added to IK waypoint target positions
        "path_noise": 0.0,
        # Probability per episode that an obstacle is spawned in the arm's path
        "obstacle_prob": 0.0,
    },
    "medium": {
        "obj_x": (0.40, 0.70),
        "obj_y": (-0.15, 0.15),
        "obj_z": (0.05, 0.15),
        "obj_yaw": (-np.pi / 6, np.pi / 6),
        "arm_jitter": 0.15,
        "path_noise": 0.015,
        "obstacle_prob": 0.0,
    },
    "high": {
        "obj_x": (0.30, 0.80),
        "obj_y": (-0.30, 0.30),
        "obj_z": (0.05, 0.25),
        "obj_yaw": (-np.pi, np.pi),
        "arm_jitter": 0.40,
        "path_noise": 0.04,
        # 50 % of episodes have an obstacle placed directly in the arm's path
        "obstacle_prob": 0.5,
    },
}

# Kuka iiwa approximate home joint angles (all zeros = straight up)
HOME_JOINTS = [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0]

# URDF for the block, we don't need to use different objects since that is not relevant here
OBJECT_URDF = "block.urdf"

# The block is 0.1m tall so its center needs to be at 0.05m for it to be resting on the plane
BLOCK_Z = 0.05

# hover height above block before descending
HOVER_HEIGHT = 0.30


class DataCollector:
    def __init__(self, variability: str, num_iterations: int,
                 steps_per_iteration: int, output_file: str):

        self.variability = variability
        self.cfg = TIER_CONFIG[variability]
        self.num_iterations = num_iterations
        self.steps_per_iteration = steps_per_iteration
        self.output_file = output_file

        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 240.0)

        self.planeId = p.loadURDF("plane.urdf")

        # Robot init
        self.kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True) # Note: we can use other robot arms too, this is just one option
        self.numJoints = p.getNumJoints(self.kukaId)
        self.joint_indices = list(range(self.numJoints))

        # Place first block and settle
        self.objectId = self._spawn_block()
        self.obstacleId = None  # managed per-episode in run()
        self._settle(30)

    def _sample_block_pose(self):
        obj_x = random.uniform(*self.cfg["obj_x"])
        obj_y = random.uniform(*self.cfg["obj_y"])
        obj_z = random.uniform(*self.cfg["obj_z"])
        yaw = random.uniform(*self.cfg["obj_yaw"])
        orientation = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        return [obj_x, obj_y, obj_z], orientation

    def _spawn_block(self):
        pos, orn = self._sample_block_pose()
        obj_id = p.loadURDF(OBJECT_URDF, pos, orn)
        return obj_id

    def _reset_block(self):
        p.removeBody(self.objectId)
        self.objectId = self._spawn_block()
        # Obstacle lifecycle is managed per-episode in run(), not here.

    def _spawn_obstacle(self, block_pos):
        """Place a sphere obstacle somewhere along the straight-line path
        from the robot base (origin) to the block.

        t in [0.3, 0.7] ensures the obstacle is in the middle section of the
        path — close enough to the robot to be in the way, but not so close
        that it sits right next to the base or the block.
        """
        t = random.uniform(0.3, 0.7)
        obs_x = t * block_pos[0] + random.gauss(0, 0.03)
        obs_y = t * block_pos[1] + random.gauss(0, 0.03)
        obs_z = 0.10  # sphere rests slightly above the ground plane
        orn = p.getQuaternionFromEuler([0, 0, 0])
        obj_id = p.loadURDF("sphere_small.urdf", [obs_x, obs_y, obs_z], orn, globalScaling=2.0)
        return obj_id

    def _reset_arm(self):
        jitter = self.cfg["arm_jitter"]
        for j_idx, home_angle in zip(self.joint_indices, HOME_JOINTS):
            angle = home_angle + random.uniform(-jitter, jitter)
            p.resetJointState(self.kukaId, j_idx, angle)
            p.resetJointState(self.kukaId, j_idx, angle, 0.0)

    def _settle(self, steps: int):
        for _ in range(steps):
            p.stepSimulation()

    def _get_block_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.objectId)
        return list(pos)

    def _add_noise(self, pos):
        # adds noise to the path
        std = self.cfg["path_noise"]
        if std == 0.0:
            return pos
        return [v + random.gauss(0, std) for v in pos]

    def _ik_joints(self, target_pos):
        # returns the joint angles to reach target_pos via PyBullet IK
        joint_angles = p.calculateInverseKinematics(
            self.kukaId,
            self.numJoints - 1,
            target_pos,
            maxNumIterations=200,
            residualThreshold=1e-5,
        )
        return list(joint_angles[: self.numJoints])

    def _set_joint_targets(self, joint_targets):
        p.setJointMotorControlArray(
            self.kukaId,
            self.joint_indices,
            p.POSITION_CONTROL,
            targetPositions=joint_targets,
            forces=[500] * self.numJoints,
        )

    def _get_joint_states(self):
        states = p.getJointStates(self.kukaId, self.joint_indices)
        positions = [s[0] for s in states]
        velocities = [s[1] for s in states]
        torques = [s[3] for s in states]
        return positions, velocities, torques

    def _get_hand_state(self):
        # Return hand position (x, y, z) and orientation (qx, qy, qz, qw). Hand = the last link of the robot
        link_state = p.getLinkState(self.kukaId, self.numJoints - 1, computeForwardKinematics=True)
        position = list(link_state[4])
        orientation = list(link_state[5])
        return position, orientation

    def _check_contact(self):
        # Outputs contact info with the TARGET object
        contacts_target = p.getContactPoints(bodyA=self.kukaId, bodyB=self.objectId)
        in_contact = 0
        total_force = 0.0
        if contacts_target and len(contacts_target) > 0:
            in_contact = 1
            total_force = sum(pt[9] for pt in contacts_target)

        # Also check obstacle contact (0 when no obstacle exists)
        obstacle_contact = 0
        obstacle_force = 0.0
        if self.obstacleId is not None:
            contacts_obs = p.getContactPoints(bodyA=self.kukaId, bodyB=self.obstacleId)
            if contacts_obs and len(contacts_obs) > 0:
                obstacle_contact = 1
                obstacle_force = sum(pt[9] for pt in contacts_obs)

        return in_contact, total_force, obstacle_contact, obstacle_force

    def _plan_waypoints(self, block_pos):
        """Compute motion waypoints for an episode.

        For low/medium (no obstacle): single direct approach to the contact point.
        For high (obstacle present): bypass waypoint around the obstacle → hover
        above block → final contact approach.
        """
        contact_pos = self._add_noise([
            block_pos[0],
            block_pos[1],
            block_pos[2] + 0.02,  # just above block surface for contact
        ])

        if self.obstacleId is None:
            # Low / medium tiers: single direct target (same behaviour as before)
            return [contact_pos]

        # --- High tier: route around the obstacle ---
        obs_pos, _ = p.getBasePositionAndOrientation(self.obstacleId)

        # Direction from robot origin toward the block
        bx, by = block_pos[0], block_pos[1]
        path_len = max(np.sqrt(bx ** 2 + by ** 2), 1e-6)

        # Perpendicular to the robot → block path (rotate 90°)
        perp_x = -by / path_len
        perp_y =  bx / path_len

        # Pick the perpendicular side that moves away from the obstacle centre
        # (dot the perpendicular with the obstacle offset; choose opposite sign)
        obs_offset_x = obs_pos[0] - bx * 0.5
        obs_offset_y = obs_pos[1] - by * 0.5
        dot = obs_offset_x * perp_x + obs_offset_y * perp_y
        side = -1.0 if dot > 0 else 1.0

        bypass_lateral = 0.30  # metres to the side of obstacle
        bypass_pos = [
            obs_pos[0] + side * perp_x * bypass_lateral,
            obs_pos[1] + side * perp_y * bypass_lateral,
            HOVER_HEIGHT + 0.10,  # safely above everything
        ]

        hover_pos = [
            block_pos[0],
            block_pos[1],
            block_pos[2] + HOVER_HEIGHT,
        ]

        return [bypass_pos, hover_pos, contact_pos]

    def _build_header(self):
        header = ["iteration", "step", "variability_level",
                  "in_contact", "contact_force",
                  "obstacle_contact", "obstacle_force"]
        for i in range(self.numJoints):
            header.extend([f"j{i}_pos", f"j{i}_vel", f"j{i}_torque"])
        header.extend(["hand_x", "hand_y", "hand_z", "hand_qx", "hand_qy", "hand_qz", "hand_qw"])
        return header

    def run(self):
        print(f"[{self.variability}] Starting data collection: {self.num_iterations} iterations x {self.steps_per_iteration} steps -> {self.output_file}")

        with open(self.output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._build_header())

            total_contacts = 0
            total_samples = 0

            for iteration in range(self.num_iterations):
                if iteration % 50 == 0:
                    pct = 100 * iteration / self.num_iterations
                    cr = (total_contacts / total_samples * 100) if total_samples > 0 else 0.0
                    print(f"  iter {iteration:>4}/{self.num_iterations}  ({pct:5.1f}%)  contact_rate={cr:.1f}%")

                # Reset arm and block
                self._reset_arm()
                self._reset_block()
                self._settle(30)

                block_pos = self._get_block_position()

                # Per-episode obstacle: remove any previous, maybe spawn a new one
                if self.obstacleId is not None:
                    p.removeBody(self.obstacleId)
                    self.obstacleId = None
                if random.random() < self.cfg["obstacle_prob"]:
                    self.obstacleId = self._spawn_obstacle(block_pos)
                    self._settle(10)  # let the obstacle settle before planning

                # Plan waypoints: direct for low/medium, bypass→hover→contact for high
                waypoints = self._plan_waypoints(block_pos)
                n_wps = len(waypoints)
                base_steps = self.steps_per_iteration // n_wps
                # Distribute any leftover steps to the last waypoint (contact phase)
                leftover = self.steps_per_iteration - base_steps * n_wps

                step = 0
                for wp_idx, wp_pos in enumerate(waypoints):
                    wp_joints = self._ik_joints(wp_pos)
                    wp_steps = base_steps + (leftover if wp_idx == n_wps - 1 else 0)

                    for _ in range(wp_steps):
                        self._set_joint_targets(wp_joints)
                        p.stepSimulation()

                        # Collect state
                        j_pos, j_vel, j_torque = self._get_joint_states()
                        hand_position, hand_orientation = self._get_hand_state()
                        in_contact, force, obs_contact, obs_force = self._check_contact()

                        row = [iteration, step, self.variability,
                               in_contact, force, obs_contact, obs_force]
                        row += j_pos + j_vel + j_torque
                        row += hand_position + hand_orientation
                        writer.writerow(row)

                        total_contacts += in_contact
                        total_samples += 1
                        step += 1

                # Flush periodically so partial results are saved on early exit
                if iteration % 10 == 0:
                    f.flush()

        contact_rate = total_contacts / total_samples if total_samples > 0 else 0.0
        print(f"[{self.variability}] Done. {total_samples} samples, contact rate: {contact_rate*100:.2f}%  -> {self.output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variability", type=str)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    collector = DataCollector(
        variability=args.variability,
        num_iterations=args.iterations,
        steps_per_iteration=args.steps,
        output_file=args.output,
    )
    collector.run()

if __name__ == "__main__":
    main()