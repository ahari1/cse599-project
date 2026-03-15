import pybullet as p
import pybullet_data
import numpy as np
import csv
import random
import argparse


TIER_CONFIG = {
    "low": {
        "obj_r": (0.45, 0.55),
        "obj_theta": (-np.pi, np.pi),
        "obj_z": (0.05, 0.05),
        "obj_yaw": (0.0, 0.0),
        "arm_jitter": 0.05,
        "path_noise": 0.0,
        "obstacle_prob": 0.0,
        "no_object_prob": 0.25,
    },
    "medium": {
        "obj_r": (0.40, 0.60),
        "obj_theta": (-np.pi, np.pi),
        "obj_z": (0.05, 0.15),
        "obj_yaw": (-np.pi / 6, np.pi / 6),
        "arm_jitter": 0.05,
        "path_noise": 0.015,
        "obstacle_prob": 0.2,
        "no_object_prob": 0.25,
    },
    "high": {
        "obj_r": (0.20, 0.80),
        "obj_theta": (-np.pi, np.pi),
        "obj_z": (0.05, 0.25),
        "obj_yaw": (-np.pi, np.pi),
        "arm_jitter": 0.05,
        "path_noise": 0.04,
        "obstacle_prob": 0.5,
        "no_object_prob": 0.25,
    },
}


HOME_JOINTS = [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0]

OBJECT_URDF = "block.urdf"

BLOCK_Z = 0.05
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

        self.kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        self.numJoints = p.getNumJoints(self.kukaId)
        self.joint_indices = list(range(self.numJoints))

        self.objectId = self._spawn_block()
        self.obstacleId = None
        self._settle(30)

    def _sample_block_pose(self):
        r = random.uniform(*self.cfg["obj_r"])
        theta = random.uniform(*self.cfg["obj_theta"])

        obj_x = r * np.cos(theta)
        obj_y = r * np.sin(theta)
        obj_z = random.uniform(*self.cfg["obj_z"])

        yaw = random.uniform(*self.cfg["obj_yaw"])
        orientation = p.getQuaternionFromEuler([0.0, 0.0, yaw])

        return [obj_x, obj_y, obj_z], orientation

    def _sample_free_target(self):
        r = random.uniform(*self.cfg["obj_r"])
        theta = random.uniform(*self.cfg["obj_theta"])

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = random.uniform(0.10, HOVER_HEIGHT)

        return self._add_noise([x, y, z])

    def _spawn_block(self):
        pos, orn = self._sample_block_pose()
        obj_id = p.loadURDF(OBJECT_URDF, pos, orn)
        return obj_id

    def _reset_block(self):
        if self.objectId is not None:
            p.removeBody(self.objectId)
        self.objectId = self._spawn_block()

    def _spawn_obstacle(self, block_pos):
        bx, by = block_pos[0], block_pos[1]

        block_r = np.sqrt(bx ** 2 + by ** 2)
        block_theta = np.arctan2(by, bx)

        t = random.uniform(0.3, 0.7)
        obs_r = t * block_r

        obs_theta = block_theta + random.uniform(-0.08, 0.08)

        obs_x = obs_r * np.cos(obs_theta)
        obs_y = obs_r * np.sin(obs_theta)
        obs_z = 0.10

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
        std = self.cfg["path_noise"]

        if std == 0.0:
            return pos

        return [v + random.gauss(0, std) for v in pos]

    def _ik_joints(self, target_pos):
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
        link_state = p.getLinkState(
            self.kukaId,
            self.numJoints - 1,
            computeForwardKinematics=True
        )

        position = list(link_state[4])
        orientation = list(link_state[5])

        return position, orientation

    def _check_contact(self):
        in_contact = 0
        total_force = 0.0

        if self.objectId is not None:
            contacts_target = p.getContactPoints(
                bodyA=self.kukaId,
                bodyB=self.objectId
            )

            if contacts_target and len(contacts_target) > 0:
                in_contact = 1
                total_force = sum(pt[9] for pt in contacts_target)

        obstacle_contact = 0
        obstacle_force = 0.0

        if self.obstacleId is not None:
            contacts_obs = p.getContactPoints(
                bodyA=self.kukaId,
                bodyB=self.obstacleId
            )

            if contacts_obs and len(contacts_obs) > 0:
                obstacle_contact = 1
                obstacle_force = sum(pt[9] for pt in contacts_obs)

        return in_contact, total_force, obstacle_contact, obstacle_force

    def _plan_waypoints(self, block_pos):
        contact_pos = self._add_noise([
            block_pos[0],
            block_pos[1],
            block_pos[2] + 0.02,
        ])

        if self.obstacleId is None:
            return [contact_pos]

        obs_pos, _ = p.getBasePositionAndOrientation(self.obstacleId)

        bx, by = block_pos[0], block_pos[1]
        path_len = max(np.sqrt(bx ** 2 + by ** 2), 1e-6)

        perp_x = -by / path_len
        perp_y = bx / path_len

        obs_offset_x = obs_pos[0] - bx * 0.5
        obs_offset_y = obs_pos[1] - by * 0.5

        dot = obs_offset_x * perp_x + obs_offset_y * perp_y

        side = -1.0 if dot > 0 else 1.0

        bypass_lateral = 0.30

        bypass_pos = [
            obs_pos[0] + side * perp_x * bypass_lateral,
            obs_pos[1] + side * perp_y * bypass_lateral,
            HOVER_HEIGHT + 0.10,
        ]

        return [bypass_pos, contact_pos]

    def _build_header(self):
        header = [
            "iteration",
            "step",
            "variability_level",
            "in_contact",
            "contact_force",
            "obstacle_contact",
            "obstacle_force"
        ]

        for i in range(self.numJoints):
            header.extend([
                f"j{i}_pos",
                f"j{i}_vel",
                f"j{i}_torque"
            ])

        header.extend([
            "hand_x",
            "hand_y",
            "hand_z",
            "hand_qx",
            "hand_qy",
            "hand_qz",
            "hand_qw"
        ])

        return header

    def run(self):
        print(
            f"[{self.variability}] Starting data collection: "
            f"{self.num_iterations} iterations x "
            f"{self.steps_per_iteration} steps "
            f"-> {self.output_file}"
        )

        with open(self.output_file, "w", newline="") as f:

            writer = csv.writer(f)
            writer.writerow(self._build_header())

            total_contacts = 0
            total_samples = 0

            for iteration in range(self.num_iterations):

                if iteration % 10 == 0:

                    pct = 100 * iteration / self.num_iterations

                    cr = (
                        total_contacts / total_samples * 100
                        if total_samples > 0 else 0.0
                    )

                    print(
                        f"  iter {iteration:>4}/{self.num_iterations} "
                        f"({pct:5.1f}%) "
                        f"contact_rate={cr:.1f}%"
                    )

                self._reset_arm()
                self._reset_block()
                self._settle(30)

                is_free_motion = random.random() < self.cfg["no_object_prob"]

                if is_free_motion:

                    if self.objectId is not None:
                        p.removeBody(self.objectId)
                        self.objectId = None

                else:
                    block_pos = self._get_block_position()

                if self.obstacleId is not None:
                    p.removeBody(self.obstacleId)
                    self.obstacleId = None

                has_obstacle = (
                    not is_free_motion
                    and random.random() < self.cfg["obstacle_prob"]
                )

                if has_obstacle:
                    self.obstacleId = self._spawn_obstacle(block_pos)
                    self._settle(10)

                if iteration % 10 == 0:

                    episode_type = (
                        "free-motion"
                        if is_free_motion else
                        ("contact+obstacle" if has_obstacle else "contact")
                    )

                    print(f"         episode_type={episode_type}")

                if is_free_motion:

                    free_target = self._sample_free_target()
                    waypoints = [free_target]

                else:

                    waypoints = self._plan_waypoints(block_pos)

                n_wps = len(waypoints)
                S = self.steps_per_iteration

                if n_wps == 2:

                    contact_steps = int(S * 0.6)
                    bypass_steps = S - contact_steps

                    wp_step_counts = [
                        bypass_steps,
                        contact_steps
                    ]

                else:

                    wp_step_counts = [S]

                step = 0

                for wp_idx, wp_pos in enumerate(waypoints):

                    wp_joints = self._ik_joints(wp_pos)
                    wp_steps = wp_step_counts[wp_idx]

                    for _ in range(wp_steps):

                        self._set_joint_targets(wp_joints)
                        p.stepSimulation()

                        j_pos, j_vel, j_torque = self._get_joint_states()

                        hand_position, hand_orientation = self._get_hand_state()

                        in_contact, force, obs_contact, obs_force = (
                            self._check_contact()
                        )

                        row = [
                            iteration,
                            step,
                            self.variability,
                            in_contact,
                            force,
                            obs_contact,
                            obs_force
                        ]

                        row += j_pos + j_vel + j_torque
                        row += hand_position + hand_orientation

                        writer.writerow(row)

                        total_contacts += in_contact
                        total_samples += 1

                        step += 1

                if iteration % 10 == 0:
                    f.flush()

        contact_rate = (
            total_contacts / total_samples
            if total_samples > 0 else 0.0
        )

        print(
            f"[{self.variability}] Done. "
            f"{total_samples} samples, "
            f"contact rate: {contact_rate*100:.2f}% "
            f"-> {self.output_file}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--variability", type=str)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    random.seed()
    np.random.seed()

    collector = DataCollector(
        variability=args.variability,
        num_iterations=args.iterations,
        steps_per_iteration=args.steps,
        output_file=args.output,
    )

    collector.run()


if __name__ == "__main__":
    main()
