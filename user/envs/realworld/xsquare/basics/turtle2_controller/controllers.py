#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" .
 .
"""
import sys
import numpy as np
import threading
import time,copy
import queue
import math
# msg
import os
import rospy
from std_msgs.msg import Float32
from arm_control.msg import JointControl,JointInformation
from arm_control.msg import PosCmd
from geometry_msgs.msg import Twist,PoseStamped
from nav_msgs.msg import Path

robot_type = os.environ.get('ROBOT_TYPE', 'TURTLE2')
if robot_type == 'TURTLE2':
    trajectory_smooth_path = "/home/arm/prj/turtle2/modules/src/trajectory_smooth"
    sys.path.append(trajectory_smooth_path)
    from src.TrajectorySmooth import TrajectorySmooth
    from turtle2_msgs_srvs.msg import HeadInfo
    from chassis_control_center.msg import LiftStatus
    # srv
    from turtle2_msgs_srvs.srv import HeadControl, HeadControlRequest, HeadControlResponse
    from chassis_control_center.srv import LiftControl, LiftControlRequest, LiftControlResponse
    from turtle2_controller.kinematics import kinematics
    from turtle2_controller.utils import quaternion_to_euler, pose_transformation
else:#
    trajectory_smooth_path = "/home/arm/prj/hybrid-robot/rosWorkspace/src/trajectory_smooth"
    sys.path.append(trajectory_smooth_path)
    from turtle2_controller.kinematics import kinematics
    from src.TrajectorySmooth import TrajectorySmooth
    from std_msgs.msg import Float64,Float32
    from turtle2_controller.utils import quaternion_to_euler, pose_transformation


class ControllerBase:
    ''' ，
    '''
    def __init__(self):
        pass
    
    def check_cmd(self,cmd):
        ''' 
        '''
        pass
    
    def send_control(self, cmd):
        ''' 
        :param args: 
        '''
        pass

    def get_data(self):
        ''' 
        :return: 
        '''
        data = None
        return data


class HeadController(ControllerBase):
    ''' 
    '''
    def __init__(self):
        self.rate = rospy.Rate(50)
        self.head_control_cli = rospy.ServiceProxy('/head/control', HeadControl)
        self.head_control_pitch_pub = rospy.Publisher('/head/control/pitch', Float32, queue_size=10)
        self.head_control_yaw_pub = rospy.Publisher('/head/control/yaw', Float32, queue_size=10)
        self.head_data = [0.0,0.0] # [pitch,yaw]
        # head_data 
        self.head_upper = [np.pi/2, np.pi/2]
        self.head_lower = [-np.pi/2, -np.pi/2]
        self.head_data_sub = rospy.Subscriber('/head/pos', HeadInfo, self.head_data_callback)

    def head_data_callback(self, data):
        ''' 
        :param data: 
        '''
        self.head_data[0] = data.pitch
        self.head_data[1] = data.yaw


    def check_cmd(self, cmd):
        ''' cmd = [pitch,yaw]
        '''
        # 
        if not isinstance(cmd, list):
            rospy.logerr("cmd must be a list")
            return False
        # 
        if len(cmd) != 2:
            rospy.logerr("cmd must be a list of length 2")
            return False
        # float
        if not isinstance(cmd[0], float) or not isinstance(cmd[1], float):
            rospy.logerr("cmd must be a list of float")
            return False
        # 
        if cmd[0] < self.head_lower[0] or cmd[0] > self.head_upper[0]:
            rospy.logerr("pitch out of range")
            return False
        if cmd[1] < self.head_lower[1] or cmd[1] > self.head_upper[1]:
            rospy.logerr("yaw out of range")
            return False
        return True
        

    def send_control(self, cmd):
        ''' , cmd = [pitch,yaw]
        TODO(marcuswu):  
        '''
        # 
        if not self.check_cmd(cmd):
            rospy.logerr("Invalid command: %s" % cmd)
            return

        req = HeadControlRequest(pitch=cmd[0], yaw=cmd[1])
        try: self.head_control_cli(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

    def send_control_pitch(self, pitch : float):
        """ 
        """
        self.head_control_pitch_pub.publish(Float32(pitch))
    
    def send_control_yaw(self, yaw : float):
        """ 
        """
        self.head_control_yaw_pub.publish(Float32(yaw))

    def get_data(self):
        ''' 
        :return: [pitch,yaw]
        '''
        return self.head_data


class PIDController:
    def __init__(self, kp=0., ki=0., kd=0.):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = None
        self.integral = 0.
        self.integral_limit = 0.3
        self.d_limit = 0.5
        self.last_derror = None
        self.alpha = 0.8
    def output(self, error):
        controller = 0.
        controller = self.kp * error
        derror = 0.
        if self.last_error is not None:
            derror = (error - self.last_error)
        
        dcontroller = derror
        if self.last_derror is not None:
            dcontroller = self.alpha * dcontroller + (1. - self.alpha) * self.last_derror
        dcontroller = max(-self.d_limit, min(dcontroller, self.d_limit))
        controller = controller + self.kd * dcontroller
        
        self.integral = self.integral + error
        self.integral = max(-self.integral_limit, min(self.integral, self.integral_limit))
        
        controller = controller + self.ki * self.integral
        self.last_error = error
        self.last_derror = derror
        return controller
    def reset(self):
        self.last_error = None
        self.last_derror = None
        self.integral = 0.


class ChassisController(ControllerBase):
    def __init__ (self):
        rospy.loginfo("ChassisController init")
        self.kinematics = kinematics()
        self.chassis_pose = [0.0,0.0,0.0,0.0,0.0,0.0,0.0] # [x,y,z,ox,oy,oz,ow]
        self.virtual_zero_tf = None
        self.virtual_zero_tf_inv = np.eye(4) # 
        self.rate = rospy.Rate(50)
        self.chassis_vel_pub = rospy.Publisher('/chassis/cmd_vel', Twist, queue_size=10)
        # 
        self.chassis_pose_sub = rospy.Subscriber('/tracked_pose', PoseStamped, self.chassis_pose_callback)
        # 
        # self.chassis_pose_tracked_pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)
        # 
        self.opimize_path_pub = rospy.Publisher('/infer_path', Path, queue_size=10)

        self.radius=0.30
        self.linear_controller_x = PIDController(1.5, 0.0, 0.001)
        self.linear_controller_y = PIDController(1.5, 0.0, 0.001)
        self.angular_controller = PIDController(1.5, 0.0, 0.)
        self.stop_moving_signal = False
        self.target_queue = queue.Queue()
        self.target_queue.maxsize = 10

        self.chassis_pose_control_thread = threading.Thread(target=self.chassis_pose_control_thread_callback)
        self.chassis_pose_control_thread.daemon = True
        self.chassis_pose_control_thread.start()

    def chassis_pose_control_thread_callback(self):
        """
        @todo:，。
        """
        target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] # [x,y,z,ox,oy,oz,ow]
        at_goal = False
        count = 0 
        while True:
            # ，
            if self.target_queue.empty(): 
                time.sleep(0.1)
                continue
            else: # ，
                at_goal = False
                # self.stop_moving_signal = False #，
                count = 0
                target = self.target_queue.get()
            if at_goal: # ，
                time.sleep(0.1)
                continue
            if self.stop_moving_signal: # 
                # 
                self.target_queue.queue.clear()
                at_goal = True
                self.stop_moving_signal = False
                time.sleep(0.1)
                continue
            
            # cpos/tpos = [x,y,yaw] 
            cpos = pose_transformation(self.chassis_pose[0:3], self.chassis_pose[3:7])
            tpos = pose_transformation(target[0:3], target[3:7])
            dx = tpos[0] - cpos[0]
            dy = tpos[1] - cpos[1]

            sin_theta_2 = self.chassis_pose[5] # orientation.z
            cos_theta_2 = self.chassis_pose[6] # orientation.w
            sin_theta = 2. * sin_theta_2 * cos_theta_2
            cos_theta = cos_theta_2 * cos_theta_2 - sin_theta_2 * sin_theta_2

            dxb = cos_theta * dx + sin_theta * dy
            dyb = -sin_theta * dx + cos_theta * dy
            dtheta = math.atan2(sin_theta, cos_theta)

            _, _, cangler = self.kinematics.quaternion_to_euler_angle(tuple(self.chassis_pose[3:7]))
            _, _, tangler = self.kinematics.quaternion_to_euler_angle(tuple(target[3:7]))
            dtheta = tangler - cangler
            if dtheta > math.pi: dtheta = dtheta - 2. * math.pi
            elif dtheta < -math.pi: dtheta = dtheta + 2. * math.pi

            twist = Twist()
            twist.linear.x = self.linear_controller_x.output(dxb)
            twist.linear.y = self.linear_controller_y.output(-dyb)
            twist.angular.z = self.angular_controller.output(-dtheta)

            error_len = dyb * dyb + dxb * dxb
            error_ang = dtheta * dtheta

            control_lenerror = 0.01 * 0.01
            control_angerr = 0.01 * 0.01
            if error_len < control_lenerror and error_ang < control_angerr:
                count = count + 1
                continue
            self.chassis_vel_pub.publish(twist)
            if count > 5: at_goal,count = True, 0

            time.sleep(0.03)
        
    def chassis_pose_callback(self, data):
        ''' 
        :param data: 
        '''
        self.chassis_pose[0] = data.pose.position.x
        self.chassis_pose[1] = data.pose.position.y
        self.chassis_pose[2] = data.pose.position.z
        self.chassis_pose[3] = data.pose.orientation.x
        self.chassis_pose[4] = data.pose.orientation.y
        self.chassis_pose[5] = data.pose.orientation.z
        self.chassis_pose[6] = data.pose.orientation.w

    def stop(self):
        self.stop_moving_signal = True

    def start(self):
        self.target_queue.queue.clear()
        self.stop_moving_signal = False
        
    def rel_pose_data(self):
        ''' 
        :return: [rel_x,rel_y, rel_yaw]
        '''
        global_tf = self.kinematics.pose_to_transformation_matrix(self.chassis_pose[0:3], self.chassis_pose[3:7])
        rel_tf = self.virtual_zero_tf_inv @ global_tf
        pos = rel_tf[0:3, 3]
        ori = self.kinematics.rotation_matrix_to_quaternion(rel_tf[0:3, 0:3])
        roll,pitch,yaw = self.kinematics.quaternion_to_euler_angle(ori)
        return [pos[0], pos[1], yaw]
    
    def pose_data(self):
        ''' ，
        :return: [x,y,yaw]
        '''
        pos = self.chassis_pose[0:3]
        roll,pitch,yaw = self.kinematics.quaternion_to_euler_angle(self.chassis_pose[3:7])
        return [pos[0], pos[1], yaw]

    def posori_data(self):
        ''' ，
        :return: [x,y,z,ox,oy,oz,ow]
        '''
        return copy.deepcopy(self.chassis_pose)

    def pose2rosmsg(self, poselist):
        ''' ROS
        :param pose: [x,y,z,ox,oy,oz,ow]
        '''
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.pose.position.x = poselist[0]
        msg.pose.position.y = poselist[1]
        msg.pose.position.z = poselist[2]
        msg.pose.orientation.x = poselist[3]
        msg.pose.orientation.y = poselist[4]
        msg.pose.orientation.z = poselist[5]
        msg.pose.orientation.w = poselist[6]
        return msg
    
    def twist2rosmsg(self, twistlist):
        ''' ROS
        :param twist: [vx,vy,vz,wx,wy,wz]
        '''
        msg = Twist()
        msg.linear.x = twistlist[0]
        msg.linear.y = twistlist[1]
        msg.linear.z = twistlist[2]
        msg.angular.x = twistlist[3]
        msg.angular.y = twistlist[4]
        msg.angular.z = twistlist[5]
        return msg

    def check_cmd(self, cmd):
        ''' 
        '''
        # 
        if not isinstance(cmd, list):
            rospy.logerr("cmd must be a list")
            return False
        # 
        if len(cmd) != 6:
            rospy.logerr("cmd must be a list of length 6")
            return False
        # float
        if not isinstance(cmd[0], float) or not isinstance(cmd[1], float) or not isinstance(cmd[2], float) or not isinstance(cmd[3], float) or not isinstance(cmd[4], float) or not isinstance(cmd[5], float):
            rospy.logerr("cmd must be a list of float")
            return False
        return True

    def send_control_vel(self,cmd):
        ''' 
        :param cmd:  cmd = [vx,vy,yaw] 
        @todo:，，。
        '''
        # if not self.check_cmd(cmd):
        #     rospy.logerr("Invalid command: %s" % cmd)
        #     return
        twist = [cmd[0], cmd[1], 0.0, 0.0, 0.0, cmd[2]]
        msg = self.twist2rosmsg(twist)
        try: self.chassis_vel_pub.publish(msg)
        except Exception as e:
            rospy.logerr("msg send failed: %s" % e)

    def send_control_global_pose(self,cmd):
        ''' 
        :param cmd:  cmd = [x,y,yaw]
        '''
        ori = self.kinematics.euler_angle_to_quaternion((0.0,0.0,cmd[2]))

        pose = [cmd[0], cmd[1], 0.0, ori[0], ori[1], ori[2], ori[3]]
        # msg = self.pose2rosmsg(pose)
        try: 
            # self.chassis_pose_tracked_pub.publish(msg)
            if self.target_queue.full(): self.target_queue.get()
            self.target_queue.put(pose)
        except Exception as e:
            rospy.logerr("msg send failed: %s" % e)

    def set_virtual_zero(self,pose):
        ''' ,
        '''
        if type(pose) != list or len(pose) != 7:
            rospy.logerr("pose must be a list of length 7")
            rospy.logerr(f"Invalid pose: {pose}")
            return
        self.virtual_zero_tf = self.kinematics.pose_to_transformation_matrix(pose[0:3],pose[3:7])
        self.virtual_zero_tf_inv = np.linalg.inv(self.virtual_zero_tf)
    
    def set_current_pose_as_virtual_zero(self):
        self.set_virtual_zero(self.chassis_pose)

    def send_control_relative_pose(self, cmd, is_arrived=False):
        ''' 
        :param cmd:  cmd = [rel_x,rel_y,rel_yaw]
        '''
        if type(self.virtual_zero_tf) != np.ndarray:
            rospy.logerr("virtual_zero_tf must be a numpy array")
            return
        pos = [cmd[0], cmd[1], 0.0]
        euler = [0.0,0.0,cmd[2]]
        relative_tf = self.kinematics.pos_euler_to_transformation_matrix(pos,euler)
        global_tf = self.virtual_zero_tf @ relative_tf
        pos = global_tf[0:3,3]
        ori = self.kinematics.rotation_matrix_to_quaternion(global_tf[0:3,0:3])
        pose = [pos[0], pos[1], 0.0, ori[0], ori[1], ori[2], ori[3]]
        
        if not is_arrived:
            try: 
                # self.chassis_pose_tracked_pub.publish(msg)
                if self.target_queue.full(): self.target_queue.get()
                self.target_queue.put(pose)
            except Exception as e:
                rospy.logerr("msg send failed: %s" % e)
        else:
            r,p,y = self.kinematics.quaternion_to_euler_angle(tuple(pose[3:7]))
            t = [pose[0],pose[1],y]
            while not self.wait_pose_arrive(t):
                if self.target_queue.full(): self.target_queue.get()
                self.target_queue.put(pose)
                time.sleep(0.1)
        return True

        
    def wait_pose_arrive(self,tpose,threshold=[0.05,0.05,0.05]):
        ''' 
        @param: tpose :  [x,y,yaw]
        :return: True if arrived, False if timeout
        '''
        cpose = self.pose_data()
        if type(tpose) != list or len(tpose) != 3:
            raise ValueError("tpose must be a list of length 3")
        if type(threshold) != list or len(threshold) != 3:
            raise ValueError("threshold must be a list of length 3")
        
        dx = tpose[0] - cpose[0]
        dy = tpose[1] - cpose[1]
        dyaw = tpose[2] - cpose[2]
        if abs(dx) < threshold[0] and abs(dy) < threshold[1] and abs(dyaw) < threshold[2]:
            return True
        return False
        
    def send_control_delta_pose(self, cmd, is_arrived = False):
        ''' 
        :param cmd:  cmd = [dx,dy,dyaw]
        '''
        tf_global = self.kinematics.pose_to_transformation_matrix(self.chassis_pose[0:3],self.chassis_pose[3:7])
        #tf_global_inv = np.linalg.inv(tf_global)
        del_pos = [cmd[0], cmd[1], 0.0]
        del_euler = [0.0, 0.0, cmd[2]]
        del_tf = self.kinematics.pos_euler_to_transformation_matrix(del_pos,del_euler)
        target_tf = tf_global @ del_tf
        target_pos, target_ori = self.kinematics.transformation_matrix_to_pos_qua(target_tf)

        pose = [target_pos[0], target_pos[1], 0.0, target_ori[0], target_ori[1], target_ori[2], target_ori[3]]
        if not is_arrived: # 
            try:
                # self.chassis_pose_tracked_pub.publish(msg)
                if self.target_queue.full(): self.target_queue.get()
                self.target_queue.put(pose)
            except Exception as e:
                rospy.logerr("msg send failed: %s" % e)
        else:
            r,p,y = self.kinematics.quaternion_to_euler_angle(tuple(pose[3:7]))
            t = [pose[0],pose[1],y]
            while not self.wait_pose_arrive(t):
                if self.target_queue.full(): self.target_queue.get()
                self.target_queue.put(pose)
                time.sleep(0.1)

    def poseList2rosmsg(self, pose_list):
        ''' ROS
        :param pose_list: [x,y,z,ox,oy,oz,ow]
        '''
        if type(pose_list) != list or len(pose_list) != 7:
            rospy.logerr("pose_list must be a list of length 7")
            return
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.pose.position.x = pose_list[0]
        msg.pose.position.y = pose_list[1]
        msg.pose.position.z = pose_list[2]
        msg.pose.orientation.x = pose_list[3]
        msg.pose.orientation.y = pose_list[4]
        msg.pose.orientation.z = pose_list[5]
        msg.pose.orientation.w = pose_list[6]
        return msg
    
    def optimize_global_path_control(self,path):
        ''' 
        @todo: ，
        param path:   [ [x,y,z,ox,oy,oz,ow], ... ]
        @warning: 
        '''
        if type(path) != list:
            rospy.logerr("path must be a list")
            return False
        # 
        if len(path) < 2:
            rospy.logerr("path must have at least 2 points")
            return False
        if len(path[0]) != 7:
            rospy.logerr("path points must be of length 7")
            return False
        # PoseStamped
        path_msg = Path()
        for p in path:
            pose_msg = self.poseList2rosmsg(p)
            path_msg.poses.append(pose_msg)
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = 'map'
        # ，
        try: self.opimize_path_pub.publish(path_msg)
        except Exception as e:
            rospy.logerr("msg send failed: %s" % e)
            return False
    
    def optimize_rel_pathxyY_control(self, path):
        ''' ,path = [[x,y,yaw], ...]
        @warning : 
        '''
        if type(path) != list:
            rospy.logerr("path must be a list")
            return False
        new_path = []
        for xyY in path:
            if len(xyY) != 3:
                rospy.logerr("path points must be of length 3")
                return False
            # 
            pos = [xyY[0], xyY[1], 0.0] # z=0.0
            qua = self.kinematics.euler_angle_to_quaternion((0.0, 0.0, xyY[2]))
            # 
            relative_tf = self.kinematics.pose_to_transformation_matrix(tuple(pos), tuple(qua))
            global_tf = self.virtual_zero_tf @ relative_tf
            pos = global_tf[0:3, 3]
            ori = self.kinematics.rotation_matrix_to_quaternion(global_tf[0:3, 0:3])
            pose = [pos[0], pos[1], 0.0, ori[0], ori[1], ori[2], ori[3]]
            new_path.append(pose)
        return self.optimize_global_path_control(new_path)


class LiftController(ControllerBase):
    def __init__(self):
        self.rate = rospy.Rate(10)
        self.lift_upper = 0.47 # 
        self.lift_lower = 0.0 # 
        self.lift_data = 0.0
        self.lift_is_cali = False
        self.lift_data_sub = rospy.Subscriber('/chassis_control_center/lift_status', LiftStatus, self.lift_data_callback)
        self.lift_control_cli = rospy.ServiceProxy('/chassis_control_center/lift_control', LiftControl)

        
    def check_cmd(self,cmd):
        ''' 
        '''
        # float
        if not isinstance(cmd, float):
            rospy.logerr("cmd must be a float")
            return False
        # 
        if cmd < self.lift_lower or cmd > self.lift_upper:
            rospy.logerr("cmd out of range")
            return False
        return True


    def lift_data_callback(self, data):
        ''' 
        :param 
        '''
        self.lift_data = data.position_m
        self.lift_is_cali = data.is_cali


    def send_control(self, cmd):
        ''' 
        :param(float) cmd 
        '''
        # 
        if not self.check_cmd(cmd):
            rospy.logerr("Invalid command: %s" % cmd)
            return

        req = LiftControlRequest(cmd=1, position=cmd) # cmd=1，。
        try: self.lift_control_cli(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)


    def get_data(self):
        ''' 
        :return(float) lift_data 
        '''
        return self.lift_data


class ArmsController(ControllerBase):
    ''' 
    '''
    def __init__(self):
        self.slave1EndPos = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.slave2EndPos = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.slave1JointCur = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.slave2JointCur = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.slave1JointPos = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.slave2JointPos = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.endPosPub1 = rospy.Publisher('/follow_pos_cmd_1', PosCmd, queue_size=5)
        self.endPosPub2 = rospy.Publisher('/follow_pos_cmd_2', PosCmd, queue_size=5)
        self.endPosSub1 = rospy.Subscriber('/follow1_pos_back', PosCmd, self.Slave1PosBack_callback)
        self.endPosSub2 = rospy.Subscriber('/follow2_pos_back', PosCmd, self.Slave2PosBack_callback)
        self.jointsSub1 = rospy.Subscriber('/joint_information', JointControl,
                                           self.Slave1JointsBack_callback)
        self.jointsSub2 = rospy.Subscriber('/joint_information2', JointControl,
                                           self.Slave2JointsBack_callback)
        self.arm_trajs = [] # [{"arml":[], "armr":[],"exit_step"=0, "t_step"=0}]
        self.trajectory_smooth = TrajectorySmooth(trajectory_smooth_path + "/config/config.yaml")
        self.arm_action_thread = None
        self.lock = threading.Lock()
        self.exit_siganl = False
        self.infer_request = False
        self.infer_time = 0.5 # 
        self.step_time = 0.005
        self.running=False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def __del__(self):
        if self.arm_action_thread is not None:
            self.exit_signal = True
            self.arm_action_thread.join()

    def Slave1PosBack_callback(self, data):
        self.slave1EndPos = [
            data.x, data.y, data.z, data.roll, data.pitch, data.yaw,data.gripper]
    
    def Slave2PosBack_callback(self, data):
        self.slave2EndPos = [
            data.x, data.y, data.z, data.roll, data.pitch, data.yaw, data.gripper]
    
    def Slave1JointsBack_callback(self, data):
        self.slave1JointPos = data.joint_pos
        self.slave1JointCur = data.joint_cur
        
    def Slave2JointsBack_callback(self, data):
        self.slave2JointPos = data.joint_pos
        self.slave2JointCur = data.joint_cur

    def pose2rosmsg(self, pose):
        ''' ROS
        :param pose: '''
        msg = PosCmd()
        msg.x = pose[0]
        msg.y = pose[1]
        msg.z = pose[2]
        msg.roll = pose[3]
        msg.pitch = pose[4]
        msg.yaw = pose[5]
        msg.gripper = pose[6]
        return msg

    def check_cmd(self,cmd1,cmd2):
        ''' 
        '''
        pass

    def send_control(self, cmd_l, cmd_r):
        ''' 
        :param cmdl&cmd_r:  [x,y,z,roll,pitch,yaw,gripper]
        '''
        msg_l = self.pose2rosmsg(cmd_l)
        self.endPosPub1.publish(msg_l)
        msg_r = self.pose2rosmsg(cmd_r)
        self.endPosPub2.publish(msg_r)
    
    def send_zero(self):
        ''' 
        '''
        pos1,pos2 = self.get_data()
        pos1[6],pos2[6] = 4.0,4.0 #
        # 400
        step = 400
        steps1 = pos1 - np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        steps2 = pos2 - np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        steps1 = steps1 / step
        steps2 = steps2 / step

        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        for i in range(step):
            self.send_control(pos1 - steps1 * i,pos2 - steps2 * i)
            time.sleep(0.005) # 200hz

        
        self.send_control([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        rospy.loginfo("send zero pose to arms")

    # 
    def send_control_linear_trj(self, target_l, target_r, steps=200):
        ''' 
        :param target_l&target_r:  [x,y,z,roll,pitch,yaw,gripper]
        :param steps: 
        :todo: ！！！！！！！！！！！！
        '''
        if len(target_l[0]) != 7 or len(target_r[0]) !=7:
            rospy.logerr("cmd_l and cmd_r points must be of length 7")
            return
        # 
        pos1, pos2 = self.get_data()
        # 
        target_l = np.array(target_l, dtype=np.float32)
        target_r = np.array(target_r, dtype=np.float32)
        if len(target_l) != len(target_r):
            rospy.logerr("target_l and target_r must have the same length")
            return
        if len(target_l) < 2:
            rospy.logerr("target_l and target_r must have at least 2 points")
            return
        # 
        pass
    
    def send_control_raw_trj(self, cmd_l : list, cmd_r : list, t_step=0.005):
        ''' 
        :param cmd_l&cmd_r:  [[x,y,z,roll,pitch,yaw,gripper],...]
        :param t_step: ，
        '''
        if len(cmd_l) != len(cmd_r):
            rospy.logerr("cmd_l and cmd_r must have the same length")
            return
        if len(cmd_l) < 2:
            rospy.logerr("cmd_l and cmd_r must have at least 2 points")
            return
        if len(cmd_l[0]) != 7 or len(cmd_r[0]) != 7:
            rospy.logerr("cmd_l and cmd_r points must be of length 7")
            return
        # 
        len_cmd = len(cmd_l)
        for i in range(len_cmd):
            self.send_control(cmd_l[i], cmd_r[i])
            time.sleep(t_step)

    def send_control_pose_trj_async(self, pose_trj_l : list, pose_trj_r : list, \
            pos_method='linear', quaternion_interpolation_method="slerp", infer_time=0.5, t_step=0.005, interpolation_step=200):
        """ 
        :param pose_trj_l&pose_trj_r: 
        :param pos_method: ，'linear' or 'toppra'
        :param quaternion_interpolation_method: ，'slerp' or 'squad'
        :param infer_time: ，
        :param t_step: ，
        :param interpolation_step: 
        @ NOTE: 。
        """
        if self.arm_action_thread is None:
            raise RuntimeError("Please start the arm action thread first by calling start_arm_actions_thread()")
        
        pose_trj_l = np.array(pose_trj_l, dtype=np.float32)
        pose_trj_r = np.array(pose_trj_r, dtype=np.float32)
        if len(pose_trj_l) != len(pose_trj_r):
            rospy.logerr("pose_trj_l and pose_trj_r must have the same length")
            return
        
        current_end_pose1, current_end_pose2 = self.get_data()
        nearest_index1 = self.trajectory_smooth.find_action_nearest_index(
            current_end_pose1, pose_trj_l[:, :3])
        nearest_index2 = self.trajectory_smooth.find_action_nearest_index(
            current_end_pose2, pose_trj_r[:, :3])
        
        if len(pose_trj_l) - nearest_index1 <=1: nearest_index1 = len(pose_trj_l) - 2
        if len(pose_trj_r) - nearest_index2 <= 1: nearest_index2 = len(pose_trj_r) - 2

        if pos_method == 'linear':
            # 
            actions1 = self.trajectory_smooth.interpolates_tcp_actions(
                pose_trj_l[nearest_index1 : ],
                target_actions_num=interpolation_step,
                quaternion_interpolation_method=quaternion_interpolation_method)
            actions2 = self.trajectory_smooth.interpolates_tcp_actions(
                pose_trj_r[nearest_index2 : ],
                target_actions_num=interpolation_step,
                quaternion_interpolation_method=quaternion_interpolation_method)
        if pos_method == 'toppra':
            nearest_index = min(nearest_index1, nearest_index2)
            actions1, actions2 = (self.trajectory_smooth.interpolation_by_toppra(
                pose_trj_l[nearest_index : ], pose_trj_r[nearest_index : ],quaternion_interpolation_method=quaternion_interpolation_method))
        
        # 
        infer_step = int((1/ t_step) * (infer_time))
        exit_step = len(actions1) - infer_step

        if infer_step < 1:
            rospy.logerr("infer_time is too small, infer_step must be greater than 1")
            return
        if exit_step < 1:
            print("-" * 30)
            exit_step = len(actions1) - 1

        print(f", : {len(actions1)}, : {exit_step}, : {infer_time}")
        self.set_arm_actions(actions1, actions2, exit_step, t_step)
        self.wait_arm_actions()


    def send_control_pose_trj_sync(self, pose_trj_l : list, pose_trj_r :list,\
             pos_method='linear', quaternion_interpolation_method="slerp" , t_step=0.005, interpolation_step=200):
        """ 
        :param pose_trj_l&pose_trj_r: . [[x,y,z,roll,pitch,yaw,gripper],...]
        :param pos_method: ，'linear' or 'toppra'
        :param quaternion_interpolation_method: ，'slerp' or 'squad'
        :param t_step: ，
        :param interpolation_step: 
        @TODO : 。
        """
        pose_trj_l = np.array(pose_trj_l, dtype=np.float32)
        pose_trj_r = np.array(pose_trj_r, dtype=np.float32)
        if len(pose_trj_l) != len(pose_trj_r):
            rospy.logerr("pose_trj_l and pose_trj_r must have the same length")
            return
        
        current_end_pose1, current_end_pose2 = self.get_data()
        nearest_index1 = self.trajectory_smooth.find_action_nearest_index(
            current_end_pose1, pose_trj_l[:, :3])
        nearest_index2 = self.trajectory_smooth.find_action_nearest_index(
            current_end_pose2, pose_trj_r[:, :3])
        
        if len(pose_trj_l) - nearest_index1 <=1: nearest_index1 = len(pose_trj_l) - 2
        if len(pose_trj_r) - nearest_index2 <= 1: nearest_index2 = len(pose_trj_r) - 2

        if pos_method == 'linear': # 
            actions1 = self.trajectory_smooth.interpolates_tcp_actions(
                pose_trj_l[nearest_index1 : ],
                target_actions_num=interpolation_step,
                quaternion_interpolation_method=quaternion_interpolation_method)
            actions2 = self.trajectory_smooth.interpolates_tcp_actions(
                pose_trj_r[nearest_index2 : ],
                target_actions_num=interpolation_step,
                quaternion_interpolation_method=quaternion_interpolation_method)
        if pos_method == 'toppra': # toppra
            nearest_index = min(nearest_index1, nearest_index2)
            actions1, actions2 = (self.trajectory_smooth.interpolation_by_toppra(
                pose_trj_l[nearest_index : ], pose_trj_r[nearest_index : ],quaternion_interpolation_method=quaternion_interpolation_method))

        # 
        for i in range(len(actions1)):
            self.send_control(actions1[i], actions2[i])
            time.sleep(t_step)

    def send_control_pose_trj(self, is_async_: bool ,pose_trj_l, pose_trj_r, \
            pos_method='linear', quaternion_interpolation_method="slerp" ,infer_time=0.5, t_step=0.005, interpolation_step=200):
        ''' 
        :param is_async_: ，True，False
        :param pose_trj_l&pose_trj_r:  [[x,y,z,roll,pitch,yaw,gripper],...]
        :param pos_method: ，'linear' or 'toppra'
        :param quaternion_interpolation_method: ，'slerp' or 'squad'
        :param infer_time: ，
        :param t_step: ，
        :param interpolation_step: 
        @TODO:  ， 1-2 
        '''
        pose_trj_l = np.array(pose_trj_l, dtype=np.float32)
        pose_trj_r = np.array(pose_trj_r, dtype=np.float32)
        if len(pose_trj_l) != len(pose_trj_r):
            rospy.logerr("pose_trj_l and pose_trj_r must have the same length")
            return

        # setp = int(len(pose_trj_l) * infer_time)
        # pose_trj_l = pose_trj_l[setp:]
        # pose_trj_r = pose_trj_r[setp:] 
        
        # 
        
        current_end_pose1, current_end_pose2 = self.get_data()
        nearest_index1 = self.trajectory_smooth.find_action_nearest_index(
            current_end_pose1, pose_trj_l[:, :3])
        nearest_index2 = self.trajectory_smooth.find_action_nearest_index(
            current_end_pose2, pose_trj_r[:, :3])
        
        if len(pose_trj_l) - nearest_index1 <=1: nearest_index1 = len(pose_trj_l) - 2
        if len(pose_trj_r) - nearest_index2 <= 1: nearest_index2 = len(pose_trj_r) - 2

        if pos_method == 'linear':
            # 
            actions1 = self.trajectory_smooth.interpolates_tcp_actions(
                pose_trj_l[nearest_index1 : ],
                target_actions_num=interpolation_step,
                quaternion_interpolation_method=quaternion_interpolation_method)
            actions2 = self.trajectory_smooth.interpolates_tcp_actions(
                pose_trj_r[nearest_index2 : ],
                target_actions_num=interpolation_step,
                quaternion_interpolation_method=quaternion_interpolation_method)
        if pos_method == 'toppra':
            nearest_index = min(nearest_index1, nearest_index2)
            actions1, actions2 = (self.trajectory_smooth.interpolation_by_toppra(
                pose_trj_l[nearest_index : ], pose_trj_r[nearest_index : ],quaternion_interpolation_method=quaternion_interpolation_method))
        
        # 
        infer_step = int((1/ t_step) * (infer_time))
        exit_step = len(actions1) - infer_step

        if infer_step < 1:
            rospy.logerr("infer_time is too small, infer_step must be greater than 1")
            return
        if exit_step < 1:
            print("-" * 30)
            exit_step = len(actions1) - 1

        if is_async_:
            print(f", : {len(actions1)}, : {exit_step}, : {infer_time}")
            self.set_arm_actions(actions1, actions2, exit_step, t_step)
            self.wait_arm_actions()
        else:
            for i in range(len(actions1)):
                if self.running == False: return
                self.send_control(actions1[i], actions2[i])
                time.sleep(t_step)

    def start_arm_actions_thread(self):
        if self.arm_action_thread is None:
            self.arm_action_thread = threading.Thread(target=self.arm_action_loop)
            self.arm_action_thread.daemon = True  # 
            self.arm_action_thread.start()

    def set_arm_actions(self, actions1, actions2, exit_step, t_step=0.005):
        actions = {"arml": actions1, "armr": actions2, "exit_step": exit_step,"t_step": t_step}
        with self.lock:
            self.arm_trajs.append(actions)
            self.infer_request = False

    def wait_arm_actions(self):
        print("...")
        while True:
            with self.lock:
                if self.infer_request: return
            time.sleep(0.1)  # ，infer_request

    def arm_action_loop(self):
        len_trajs = 0
        while len_trajs == 0: 
            with self.lock:
                len_trajs = len(self.arm_trajs)
            time.sleep(0.1)  # ，arm_trajs
        # 
        print("")
        with self.lock: traj = self.arm_trajs.pop(0)
        actions1 = list(traj["arml"])  # ，
        actions2 = list(traj["armr"])  # ，
        exit_step = traj["exit_step"] #
        t_step = traj["t_step"]
        count = 0
        
        while not self.exit_siganl:
            # 
            while len(actions1) > 0:
                if count >= exit_step:
                    self.infer_request = True
                # print(f"actions1: {actions1}, actions2: {actions2}")
                self.send_control(actions1.pop(0), actions2.pop(0))
                count += 1
                time.sleep(t_step)

            # ，
            len_trajs = 0
            # 
            while len_trajs == 0:
                print("...")
                with self.lock: len_trajs = len(self.arm_trajs)
                time.sleep(0.1)  # ，arm_trajs
            # 
            with self.lock: traj = self.arm_trajs.pop(0)
            actions1 = list(traj["arml"])
            actions2 = list(traj["armr"])
            exit_step = traj["exit_step"] #
            t_step = traj["t_step"]
            count = 0

            time.sleep(0.1)

    def get_data(self):
        return self.slave1EndPos, self.slave2EndPos

    def get_cur_data(self):
        return self.slave1JointCur, self.slave2JointCur
    
    def get_pos_data(self):
        return self.slave1JointPos, self.slave2JointPos


#
class MovingHeadController(ControllerBase):
    ''' 
    '''
    def __init__(self):
        self.head_pitch_sub = rospy.Subscriber('/head_pitch', Float64,
                                           self.head_pitch_cb)
        self.head_yaw_sub = rospy.Subscriber('/head_yaw', Float64,
                                           self.head_yaw_cb)

        self.head_data = [0.0,0.0] # [pitch,yaw]
        # head_data 
        self.head_upper = [np.pi/2, np.pi/2]
        self.head_lower = [-np.pi/2, -np.pi/2]

    def head_pitch_cb(self,data):
        self.head_data[0]=data.data
    def head_yaw_cb(self,data):
        self.head_data[1]=data.data

    def check_cmd(self, cmd):
        ''' cmd = [pitch,yaw]
        '''
        # 
        if not isinstance(cmd, list):
            rospy.logerr("cmd must be a list")
            return False
        # 
        if len(cmd) != 2:
            rospy.logerr("cmd must be a list of length 2")
            return False
        # float
        if not isinstance(cmd[0], float) or not isinstance(cmd[1], float):
            rospy.logerr("cmd must be a list of float")
            return False
        # 
        if cmd[0] < self.head_lower[0] or cmd[0] > self.head_upper[0]:
            rospy.logerr("pitch out of range")
            return False
        if cmd[1] < self.head_lower[1] or cmd[1] > self.head_upper[1]:
            rospy.logerr("yaw out of range")
            return False
        return True
        

    def send_control(self, cmd):
        ''' , cmd = [pitch,yaw]
        '''
        # 
        if not self.check_cmd(cmd):
            rospy.logerr("Invalid command: %s" % cmd)
            return
        rospy.set_param('/angle_p', cmd[0])
        rospy.set_param('/angle_y', cmd[1])


    def get_data(self):
        ''' 
        :return: [pitch,yaw]
        '''
        return self.head_data


class MovingLiftController(ControllerBase):
    def __init__(self):
        self.rate = rospy.Rate(10)
        self.lift_upper = 0.47 # 
        self.lift_lower = 0.0 # 
        self.lift_data = 0.0
        self.lift_sub=rospy.Subscriber('/lifting_mechanism_position',Float32,self.lift_data_callback)

        
    def check_cmd(self,cmd):
        ''' 
        '''
        # float
        if not isinstance(cmd, float):
            rospy.logerr("cmd must be a float")
            return False
        # 
        if cmd < self.lift_lower or cmd > self.lift_upper:
            rospy.logerr("cmd out of range")
            return False
        return True


    def lift_data_callback(self, data):
        ''' 
        :param 
        '''
        self.lift_data = data.data


    def send_control(self, cmd):
        ''' 
        :param(float) cmd 
        '''
        # 
        if not self.check_cmd(cmd):
            rospy.logerr("Invalid command: %s" % cmd)
            return
        
        rospy.set_param("position", cmd)



    def get_data(self):
        ''' 
        :return(float) lift_data 
        '''
        return self.lift_data


class MovingChassisController(ChassisController):
    def __init__ (self):
        self.kinematics = kinematics()
        self.chassis_pose = [0.0,0.0,0.0,0.0,0.0,0.0,0.0] # [x,y,z,ox,oy,oz,ow]
        self.virtual_zero_tf = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.virtual_zero_tf_inv = np.eye(4) # 
        self.rate = rospy.Rate(50)
        self.chassis_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # 
        self.chassis_pose_sub = rospy.Subscriber('/tracked_pose', PoseStamped, self.chassis_pose_callback)
        # 
        # self.chassis_pose_tracked_pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)
        # 
        self.opimize_path_pub = rospy.Publisher('/infer_path', Path, queue_size=10)

        self.radius=0.30
        self.linear_controller_x = PIDController(0.25, 0.0, 0.001)
        self.linear_controller_y = PIDController(0.25, 0.0, 0.001)
        self.angular_controller = PIDController(0.25, 0.0, 0.)
        self.stop_moving_signal = False
        self.target_queue = queue.Queue()
        self.target_queue.maxsize = 10

        self.chassis_pose_control_thread = threading.Thread(target=self.chassis_pose_control_thread_callback)
        self.chassis_pose_control_thread.daemon = True
        self.chassis_pose_control_thread.start()

