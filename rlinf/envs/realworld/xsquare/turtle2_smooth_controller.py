#!/usr/bin/env python3
import rospy
import sys
import os
# import rospkg
# rospack = rospkg.RosPack()
# package_path = rospack.get_path('turtle2_controller')
# sys.path.append(os.path.join(rospack_path, 'turtle2_controller'))
from turtle2_controller.Turtle2Controller import Turtle2Controller

import json
import socket
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
import struct

from rlinf.envs.realworld.common.ros import ROSController
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger
from .turtle2_robot_state import Turtle2RobotState


class Turtle2SmoothController(Worker):
    """Controller for turtle2 robot, XSquare"""

    @staticmethod
    def launch_controller(
        freq: int = 50,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        ros_pkg: str = "serl_franka_controllers",
    ):
        """Launch a FrankaController on the specified worker's node.

        Args:
            robot_ip (str): The IP address of the robot arm.
            env_idx (int): The index of the environment.
            node_rank (int): The rank of the node to launch the controller on.
            worker_rank (int): The rank of the env worker to the controller is associated with.
            ros_pkg (str): The ROS package name for the Franka controllers.

        Returns:
            FrankaController: The launched FrankaController instance.
        """
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return Turtle2SmoothController.create_group(freq).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"Turtle2SmoothController-{worker_rank}-{env_idx}",
        )

    def __init__(self, freq = 50):
        super().__init__()
        self._logger = get_logger()
        # FIXME: should move to roscontroller
        rospy.init_node("Turtle2_Smooth_Controller_Node")
        self.bridge = CvBridge()
        # FIXME: should rewrite with roscontroller
        self.controller = Turtle2Controller()

        self.controller.chassis_set_current_pose_as_virtual_zero()

        self._state = Turtle2RobotState()

        control_period = rospy.Duration(1/freq)
        state_period = rospy.Duration(0.5/freq)

        self.left_arm_target = [0, 0, 0, 0, 0, 0, 0]
        self.right_arm_target = [0, 0, 0, 0, 0, 0, 0]

        # xyz, rpy, gripper
        self.tol = [0.007, 0.02, 5] # m, rad, cm
        self.xyz_speed = 0.8  # m/s
        self.rpy_speed = 2.5  # rad/s
        self.freq = freq

        # FIXME: should move to roscontroller
        rospy.Timer(control_period, self.smooth_action_callback)
        rospy.Timer(state_period, self.state_callback)


    def state_callback(self, event):
        arms_data = self.controller.arms_data()
        self._state.follow1_pos = np.array(arms_data[0], dtype=np.float32)
        self._state.follow2_pos = np.array(arms_data[1], dtype=np.float32)
        joint_data = self.controller.arms_joint_data()
        self._state.follow1_joints = np.array(joint_data[0], dtype=np.float32)
        self._state.follow2_joints = np.array(joint_data[1], dtype=np.float32)
        cur_data = self.controller.arms_cur_data()
        self._state.follow1_cur_data = np.array(cur_data[0], dtype=np.float32)
        self._state.follow2_cur_data = np.array(cur_data[1], dtype=np.float32)
        head_data = self.controller.head_data()
        self._state.head_pos = np.array(head_data, dtype=np.float32)
        self._state.lift = float(self.controller.lift_data())
        chassis_pose = self.controller.chassis_pose_data()
        self._state.car_pose = np.array(chassis_pose, dtype=np.float32)

    def get_state(self):
        return self._state

    def smooth_action_callback(self, event):
        # print("intimer")
        xyz_step = self.xyz_speed / self.freq  # m
        rpy_step = self.rpy_speed / self.freq  # rad
        # start_time = time.time()
        
        curxyz1 = self._state.follow1_pos[0:3]
        curxyz2 = self._state.follow2_pos[0:3]
        # print("current pos:")
        # print(curxyz1, curxyz2)
        targetxyz1 = np.array(self.left_arm_target[0:3], dtype=float)
        targetxyz2 = np.array(self.right_arm_target[0:3], dtype=float)
        # print("target pos:")
        # print(targetxyz1, targetxyz2)
        errxyz1 = np.linalg.norm(curxyz1 - targetxyz1)
        errxyz2 = np.linalg.norm(curxyz2 - targetxyz2)

        currpy1 = self._state.follow1_pos[3:6]
        currpy2 = self._state.follow2_pos[3:6]
        # print("current rpy:")
        # print(currpy1, currpy2)
        targetrpy1 = np.array(self.left_arm_target[3:6], dtype=float)
        targetrpy2 = np.array(self.right_arm_target[3:6], dtype=float)
        # print("target rpy:")
        # print(targetrpy1, targetrpy2)
        errrpy1 = np.linalg.norm(currpy1 - targetrpy1)
        errrpy2 = np.linalg.norm(currpy2 - targetrpy2)
        
        if errxyz1 < self.tol[0] and errxyz2 < self.tol[0] and errrpy1 < self.tol[1] and errrpy2 < self.tol[1]:
            # print(f"[INFO] target reach! {errxyz1:.4f}, {errxyz2:.4f}, {errrpy1:.4f}, {errrpy2:.4f}")
            return
        else:
            # interpolate xyz
            dirxyz1 = (targetxyz1 - curxyz1) / (errxyz1 + 0.005)
            dirxyz2 = (targetxyz2 - curxyz2) / (errxyz2 + 0.005)
            # print("dirxyz2:",dirxyz2)
            stepxyz1 = dirxyz1 * min(xyz_step, errxyz1)
            stepxyz2 = dirxyz2 * min(xyz_step, errxyz2)
            # print("stepxyz2:",stepxyz2)
            newxyz1 = curxyz1 + stepxyz1
            newxyz2 = curxyz2 + stepxyz2
            
            # interpolate rpy
            dirrpy1 = (targetrpy1 - currpy1) / (errrpy1 + 0.005)
            dirrpy2 = (targetrpy2 - currpy2) / (errrpy2 + 0.005)
            # print("dirrpy2:",dirrpy2)
            steprpy1 = dirrpy1 * min(rpy_step, errrpy1)
            steprpy2 = dirrpy2 * min(rpy_step, errrpy2)
            newrpy1 = currpy1 + steprpy1
            newrpy2 = currpy2 + steprpy2
            # print("steprpy2:",steprpy2)
            newpos1 = [newxyz1[0], newxyz1[1], newxyz1[2], newrpy1[0], newrpy1[1], newrpy1[2], self.left_arm_target[6]]
            newpos2 = [newxyz2[0], newxyz2[1], newxyz2[2], newrpy2[0], newrpy2[1], newrpy2[2], self.right_arm_target[6]]
            # print("new pos:",newpos2)
            self.controller.arms_control(newpos1, newpos2)
            # time.sleep(0.2 / self.freq)

    def move_arm(self, left_arm_target, right_arm_target):
        assert isinstance(left_arm_target, list) and len(left_arm_target) ==7, "left_arm_target should be a list of length 7"
        assert isinstance(right_arm_target, list) and len(right_arm_target) ==7, "right_arm_target should be a list of length 7"
        self.left_arm_target = left_arm_target
        self.right_arm_target = right_arm_target
        print("Set new target:")
        print(self.left_arm_target)
        print(self.right_arm_target)

    def reset_arms(self):
        self.left_arm_target = [0, 0, 0, 0, 0, 0, 0]
        self.right_arm_target = [0, 0, 0, 0, 0, 0, 0]
        print("Reset target to zero.")
        time.sleep(2.0)