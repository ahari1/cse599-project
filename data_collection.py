import pybullet as p
import pybullet_data
import numpy as np
import csv
import random
import argparse


class DataCollector:
    OBJECT_URDFS = [
        "cube_small.urdf",
        "sphere_small.urdf",
        "block.urdf",
    ]

    def __init__(self, num_iterations, steps_per_iteration, output_file):
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

        # choose a random object and position
        urdf = random.choice(self.OBJECT_URDFS)
        obj_x = random.uniform(0.3, 0.8)
        obj_y = random.uniform(-0.3, 0.3)
        obj_z = 0.1
        orientation  = p.getQuaternionFromEuler([0, 0, random.uniform(0, 2 * np.pi)])
        self.objectId = p.loadURDF(urdf, [obj_x, obj_y, obj_z], orientation)
        for i in range(50):
            p.stepSimulation()

    def _reset_iteration(self):
        """Resets setup for next iteration"""
        
        p.removeBody(self.objectId)

        # Randomize object type and position
        urdf = random.choice(self.OBJECT_URDFS)
        obj_x = random.uniform(0.3, 0.8)
        obj_y = random.uniform(-0.3, 0.3)
        obj_z = 0.1
        orientation  = p.getQuaternionFromEuler([0, 0, random.uniform(0, 2 * np.pi)])

        self.objectId = p.loadURDF(urdf, [obj_x, obj_y, obj_z], orientation)

        for j in self.joint_indices:
            start_pos = random.uniform(-0.5, 0.5)
            p.resetJointState(self.kukaId, j, start_pos)

        for i in range(50):
            p.stepSimulation()

    def _get_joint_states(self):
        # Return robot joint telemetries
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
        # Outputs if there is a contact at the current point and if so, the amount of force
        contacts = p.getContactPoints(bodyA=self.kukaId, bodyB=self.objectId)
        in_contact = 0
        total_force = 0.0
        if len(contacts) > 0:
            in_contact = 1
            total_force = sum(pt[9] for pt in contacts)
        return in_contact, total_force

    def _build_header(self):
        # Format for the csv data file
        header = ["iteration", "step", "in_contact", "contact_force"]
        for i in range(self.numJoints):
            header.extend([f"j{i}_pos", f"j{i}_vel", f"j{i}_torque"])
        header.extend(["hand_x", "hand_y", "hand_z", "hand_qx", "hand_qy", "hand_qz", "hand_qw"])
        return header

    def run(self):
        """Runs data collection """
        print("Start data coll")

        with open(self.output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._build_header())

            total_contacts = 0
            total_samples = 0

            for iteration in range(self.num_iterations):
                
                # One random target per iteration
                target_pos = [random.uniform(-1.5, 1.5) for _ in range(self.numJoints)]
                p.setJointMotorControlArray(
                    self.kukaId,
                    self.joint_indices,
                    p.POSITION_CONTROL,
                    targetPositions=target_pos,
                    forces=[500] * self.numJoints,
                )

                for step in range(self.steps_per_iteration):
                    p.stepSimulation()

                    # Collect state
                    j_pos, j_vel, j_torque = self._get_joint_states()
                    hand_position, hand_orientation = self._get_hand_state()
                    in_contact, force = self._check_contact()

                    row = [iteration, step, in_contact, force]
                    row += j_pos + j_vel + j_torque
                    row += hand_position + hand_orientation
                    writer.writerow(row)

                    total_contacts += in_contact
                    total_samples += 1

                self._reset_iteration()

        print("Data coll done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=str, default="data.csv")
    args = parser.parse_args()

    collector = DataCollector(
        num_iterations=args.iterations,
        steps_per_iteration=args.steps,
        output_file=args.output,
    )
    collector.run()

if __name__ == "__main__":
    main()