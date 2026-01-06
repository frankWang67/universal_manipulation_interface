# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import time
import numpy as np

from multiprocessing.managers import SharedMemoryManager
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.real_world.robotiq_controller import RobotiqController

if __name__ == "__main__":
    robot_ip = "192.168.54.130"

    with SharedMemoryManager() as shm_manager:
        with RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            verbose=False
        ) as controller, \
        RobotiqController(
            shm_manager=shm_manager,
            verbose=False
        ) as gripper:
            print("Controller started.")
            state = controller.get_state()
            tcp_pose = state['ActualTCPPose']
            print("Current TCP Pose via controller:", tcp_pose)
            # print("Current TCP Pose via API:", controller.rtde_r.getActualTCPPose())

            # Move to a new position
            new_pose = tcp_pose + np.array([0.1, 0, 0, 0, 0, 0])
            controller.servoL(new_pose, duration=2.0)
            time.sleep(2.5)  # wait for motion to complete

            state = controller.get_state()
            tcp_pose = state['ActualTCPPose']
            print("New TCP Pose via controller:", tcp_pose)
            # print("New TCP Pose via API:", controller.rtde_r.getActualTCPPose())

            # print("Testing gripper...")
            # gripper.schedule_waypoint(target_pos=0.0, target_time=0.1 + time.time())  # Close gripper
            # print("Gripper closing...")
            # time.sleep(2.5)  # wait for gripper action to complete