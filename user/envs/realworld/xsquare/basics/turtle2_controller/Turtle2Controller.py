import numpy as np
import rospy
from turtle2_controller.kinematics import kinematics
from turtle2_controller.controllers import HeadController,ChassisController,LiftController,ArmsController,MovingHeadController,MovingChassisController,MovingLiftController
from turtle2_controller.sensors import Camera
import time
'''
IO,，。
'''

class RobotController:
    ''' ，
    '''
    def __init__(self):
        pass

    def head_control(self, cmd):
        ''' 
        :param cmd : 
        '''
        pass
    def head_data(self):
        ''' 
        :return: [pitch,yaw]
        '''
        pass
    def lift_control(self, cmd):
        ''' 
        :param lift_cmd: 
        '''
        pass
    def lift_data(self):
        ''' 
        :return: 
        '''
        pass
    def chassis_control(self, cmd):
        ''' 
        :param chassis_cmd: 
        '''
        pass
    def chassis_data(self):
        ''' 
        :return: 
        '''
        pass
    def get_infer_flag(self):
        infer_flag=rospy.get_param("/master_teach_mode",0)
        return (infer_flag==0)

 
class Turtle2Controller(RobotController):
    ''' 2
    ： ， c = Turtle2Controller()
     c.xxx_data() # ，
     c.xxx_control(cmd) # cmd,
    '''
    def __init__(self,init_node=False):
        if init_node:
            rospy.init_node('turtle2_controller')
        self.data_yaw = 0
        self.data_pitch = 0
        self.kinematics =  kinematics()
        self.head = HeadController()
        self.lift = LiftController()
        self.arms = ArmsController()
        self.chassis = ChassisController()
        self.cam = Camera()
        rospy.Rate(2).sleep() # 

        self.virtual_zero_tf_inv = None

    def head_control(self, cmd):
        ''' 
        :param cmd : 
        '''
        # self.head.send_control(cmd)
        self.head.send_control_pitch(cmd[0])
        self.head.send_control_yaw(cmd[1])
        
    def head_control_pitch(self, pitch : float):
        """ 
        """
        self.head.send_control_pitch(pitch)
    
    def head_control_yaw(self, yaw : float):
        """ 
        """
        self.head.send_control_yaw(yaw)

    def head_data(self):
        ''' 
        :return: list([pitch,yaw])
        '''
        return self.head.get_data()

    def lift_control(self,cmd):
        ''' 
        :param lift_cmd: 
        '''
        self.lift.send_control(cmd)

    def lift_data(self):
        ''' 
        :return: 
        '''
        return self.lift.get_data()
    
    def arms_start(self):
        """ 
        """
        self.arms.start()


    def arms_stop(self):
        """ 
        """
        self.arms.stop()

    
    def arms_control(self,cmd_l,cmd_r):
        ''' 
        :param arm_l&arm_r:  [x,y,z,roll,pitch,yaw,gripper]
        '''
        self.arms.send_control(cmd_l,cmd_r)

    def arms_control_pose_trj(self, is_async_: bool, pose_trj_l, pose_trj_r, pos_method='linear', quaternion_interpolation_method="slerp" ,infer_time=0.05, step_time=0.005, interpolation_step=200):
        ''' 
        :param pose_trj_l&pose_trj_r:  [[x,y,z,roll,pitch,yaw,gripper],...]
        :method : ,'slerp''toppra'
        '''
        self.arms.send_control_pose_trj(is_async_, pose_trj_l, pose_trj_r, pos_method, quaternion_interpolation_method, infer_time, step_time, interpolation_step)

    def arms_control_raw_trj(self,cmd_l, cmd_r,t_step=0.005):
        """ 
        :param cmd_l:  [[x,y,z,roll,pitch,yaw,gripper],...]
        :param cmd_r:  [[x,y,z,roll,pitch,yaw,gripper],...]
        :param t_step: 
        ，。，。
        """
        self.arms.send_control_raw_trj(cmd_l, cmd_r, t_step)

    def start_arm_actions_thread(self):
        self.arms.start_arm_actions_thread()

    def arms_data(self):
        ''' 
        :return:  arm_l[x,y,z,roll,pitch,yaw,gripper], arm_r[x,y,z,roll,pitch,yaw,gripper]
        '''
        return self.arms.get_data() # left_arm, right_arm
    
    def arms_cur_data(self):
        ''' '''
        return self.arms.get_cur_data()
    
    def arms_joint_data(self):
        ''' 
        :return: arm_l[joint1,joint2,joint3,joint4,joint5,joint6], arm_r[joint1,joint2,joint3,joint4,joint5,joint6]
        '''
        return self.arms.get_pos_data()

    def arms_zero(self):
        ''' 
        '''
        self.arms.send_zero()
    
    def arms_zero_after_oepen_gripper(self):
        ''' 
        '''
        arm_l,arm_r = self.arms_data()
        arm_l[6] = 4.0
        arm_r[6] = 4.0
        self.arms.send_control(arm_l,arm_r)
        rospy.Rate(0.5).sleep()
        arm_l = [0.0,0.0,0.0,0.0,0.0,0.0,4.0]
        arm_r = [0.0,0.0,0.0,0.0,0.0,0.0,4.0]
        self.arms.send_control(arm_l,arm_r)
        rospy.Rate(0.5).sleep()
        arm_l = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        arm_r = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.arms.send_control(arm_l,arm_r)

    def chassis_control_vel(self,cmd):
        ''' 
        :param cmd:  [vx,vy,dyaw]
        '''
        self.chassis.send_control_vel(cmd)

    def chassis_pose_data(self):
        ''' 
        :return:  [x,y,yaw]
        '''
        return self.chassis.pose_data()
    
    def chassis_rel_pose_data(self):
        ''' 
        :return:  [x,y,yaw]
        '''
        return self.chassis.rel_pose_data()
    
    def chassis_control_global_pose(self,cmd):
        ''' 
        :param cmd:  [x,y,yaw]
        '''
        self.chassis.send_control_global_pose(cmd)

    def chassis_set_virtual_zero(self,pose):
        ''' ,
        :param pose:  [x,y,yaw]
        @todo :
        '''
        self.chassis.set_virtual_zero(pose)

    def chassis_set_current_pose_as_virtual_zero(self):
        ''' ,
        '''
        self.chassis.set_current_pose_as_virtual_zero()

    def chassis_stop(self):
        """ 
        ，
        """
        self.chassis.stop()

    def chassis_start(self):
        """ 
        ，
        """
        self.chassis.start()

    def chassis_control_relative_pose(self, cmd, is_arrived=False):
        ''' 
        @param cmd:  [x,y,yaw]
        @param is_arrived: ，False
        '''
        return self.chassis.send_control_relative_pose(cmd, is_arrived)

    def chassis_control_delta_pose(self,cmd,is_arrived=False):
        ''' 
        :param cmd:  [x,y,yaw]
        '''
        self.chassis.send_control_delta_pose(cmd, is_arrived)

    def chassis_move(self,cmd,t):
        ''' warning: use this function carefully
        :param cmd: cmd=[px,py,yaw]
        :param cmd: t = <float>
        '''
        st = time.time()
        while time.time() - st < t:
            self.chassis.send_control_vel([i/t for i in cmd])
            time.sleep(0.02)
    
    def optimize_rel_pathxyY_control(self, path):
        ''' 
        :param path:  [[x,y,yaw],...]
        '''
        self.chassis.optimize_pathxyY_control(path)

    def cam1_data(self):
       return self.cam.get_cam1_data()
    
    def cam2_data(self):
        return self.cam.get_cam2_data()

    def cam3_data(self):
        return self.cam.get_cam3_data()
    
    # lambdaJPEG
    def cam1_compress(self):
        return self.cam.compress_image(self.cam.get_cam1_data())
    
    def cam2_compress(self):
        return self.cam.compress_image(self.cam.get_cam2_data())
    
    def cam3_compress(self):
        return self.cam.compress_image(self.cam.get_cam3_data())


class MovingController(RobotController):
    ''' 2
    ： ， c = MovingController()
     c.xxx_data() # ，
     c.xxx_control(cmd) # cmd,
    '''
    def __init__(self,init_node=False):
        if init_node:
            rospy.init_node('moving_controller')
        self.data_yaw = 0
        self.data_pitch = 0
        self.kinematics =  kinematics()
        self.head = MovingHeadController()
        self.lift = MovingLiftController()
        self.arms = ArmsController()
        self.chassis = MovingChassisController()
        self.cam = Camera()
        rospy.Rate(2).sleep() # 

        self.virtual_zero_tf_inv = None

    def head_control(self, cmd):
        ''' 
        :param cmd : 
        '''
        self.head.send_control(cmd)

    def head_data(self):
        ''' 
        :return: list([pitch,yaw])
        '''
        return self.head.get_data()

    def lift_control(self,cmd):
        ''' 
        :param lift_cmd: 
        '''
        self.lift.send_control(cmd)

    def lift_data(self):
        ''' 
        :return: 
        '''
        return self.lift.get_data()
    
    def arms_start(self):
        """ 
        """
        self.arms.start()


    def arms_stop(self):
        """ 
        """
        self.arms.stop()

    def arms_control(self,cmd_l,cmd_r):
        ''' 
        :param arm_l&arm_r:  [x,y,z,roll,pitch,yaw,gripper]
        '''
        self.arms.send_control(cmd_l,cmd_r)

    def arms_control_pose_trj(self, is_async_: bool, pose_trj_l, pose_trj_r, \
                pos_method='linear', quaternion_interpolation_method="slerp" ,infer_time=0.05, step_time=0.005, interpolation_step=200):
        ''' 
        :param pose_trj_l&pose_trj_r:  [[x,y,z,roll,pitch,yaw,gripper],...]
        :param pos_method: ,'linear''toppra'
        :param quaternion_interpolation_method: ,'slerp''squad'
        :param infer_time: ，
        :param step_time: ，
        :param interpolation_step: 
        :method : ,'slerp''toppra'
        @TODO:  ， 1-2  
        '''
        self.arms.send_control_pose_trj(is_async_, pose_trj_l, pose_trj_r, pos_method, quaternion_interpolation_method, infer_time, step_time, interpolation_step)

    def arms_control_pose_trj_sync(self, pose_trj_l, pose_trj_r, \
                pos_method='linear', quaternion_interpolation_method="slerp", t_step=0.005, interpolation_step=200):
        ''' ()
        :param pose_trj_l&pose_trj_r:  [[x,y,z,roll,pitch,yaw,gripper],...]
        :param pos_method: ,'linear''toppra'
        :param quaternion_interpolation_method: ,'slerp''squad'
        :param t_step: ，
        :param interpolation_step: 
        :method : ,'slerp''toppra'
        '''
        self.arms.send_control_pose_trj_sync(pose_trj_l, pose_trj_r, pos_method, quaternion_interpolation_method, t_step, interpolation_step)
    
    def arms_control_pose_trj_async(self, pose_trj_l, pose_trj_r, \
                pos_method='linear', quaternion_interpolation_method="slerp" ,infer_time=0.05, t_step=0.005, interpolation_step=200):
        ''' ()
        :param pose_trj_l&pose_trj_r:  [[x,y,z,roll,pitch,yaw,gripper],...]
        :param pos_method: ,'linear''toppra'
        :param quaternion_interpolation_method: ,'slerp''squad'
        :param infer_time: ，
        :param t_step: ，
        :param interpolation_step: 
        :method : ,'slerp''toppra'
        @ NOTE: 。
        '''
        self.arms.send_control_pose_trj_async(pose_trj_l, pose_trj_r, pos_method, quaternion_interpolation_method, infer_time, t_step, interpolation_step)

    def arms_control_raw_trj(self,cmd_l, cmd_r,t_step=0.005):
        """ 
        :param cmd_l:  [[x,y,z,roll,pitch,yaw,gripper],...]
        :param cmd_r:  [[x,y,z,roll,pitch,yaw,gripper],...]
        :param t_step: 
        ，。，。
        """
        self.arms.send_control_raw_trj(cmd_l, cmd_r, t_step)
    def start_arm_actions_thread(self):
        self.arms.start_arm_actions_thread()

    def arms_data(self):
        ''' 
        :return:  arm_l[x,y,z,roll,pitch,yaw,gripper], arm_r[x,y,z,roll,pitch,yaw,gripper]
        '''
        return self.arms.get_data() # left_arm, right_arm
    
    def arms_cur_data(self):
        ''' '''
        return self.arms.get_cur_data()
    
    def arms_joint_data(self):
        ''' 
        :return: arm_l[joint1,joint2,joint3,joint4,joint5,joint6], arm_r[joint1,joint2,joint3,joint4,joint5,joint6]
        '''
        return self.arms.get_pos_data()

    def arms_zero(self):
        ''' 
        '''
        self.arms.send_zero()
    
    def arms_zero_after_oepen_gripper(self):
        ''' 
        '''
        arm_l,arm_r = self.arms_data()
        arm_l[6] = 4.0
        arm_r[6] = 4.0
        self.arms.send_control(arm_l,arm_r)
        rospy.Rate(0.5).sleep()
        arm_l = [0.0,0.0,0.0,0.0,0.0,0.0,4.0]
        arm_r = [0.0,0.0,0.0,0.0,0.0,0.0,4.0]
        self.arms.send_control(arm_l,arm_r)
        rospy.Rate(0.5).sleep()
        arm_l = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        arm_r = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.arms.send_control(arm_l,arm_r)

    def chassis_control_vel(self,cmd):
        ''' 
        :param cmd:  [vx,vy,dyaw]
        '''
        self.chassis.send_control_vel(cmd)

    def chassis_pose_data(self):
        ''' 
        :return:  [x,y,yaw]
        '''
        return self.chassis.pose_data()
    
    def chassis_rel_pose_data(self):
        ''' 
        :return:  [x,y,yaw]
        '''
        return self.chassis.rel_pose_data()
    
    def chassis_control_global_pose(self,cmd):
        ''' 
        :param cmd:  [x,y,yaw]
        '''
        self.chassis.send_control_global_pose(cmd)

    def chassis_set_virtual_zero(self,pose):
        ''' ,
        :param pose:  [x,y,yaw]
        @todo :
        '''
        self.chassis.set_virtual_zero(pose)

    def chassis_set_current_pose_as_virtual_zero(self):
        ''' ,
        '''
        self.chassis.set_current_pose_as_virtual_zero()

    def chassis_stop(self):
        """ 
        ，
        """
        self.chassis.stop()

    def chassis_start(self):
        """ 
        ，
        """
        self.chassis.start()

    def chassis_control_relative_pose(self,cmd,is_arrived=False):
        ''' 
        :param cmd:  [x,y,yaw]
        '''
        return self.chassis.send_control_relative_pose(cmd,is_arrived)

    def chassis_control_delta_pose(self,cmd):
        ''' 
        :param cmd:  [x,y,yaw]
        '''
        self.chassis.send_control_delta_pose(cmd)

    def chassis_move(self,cmd,t):
        ''' warning: use this function carefully
        :param cmd: cmd=[px,py,yaw]
        :param cmd: t = <float>
        '''
        st = time.time()
        while time.time() - st < t:
            self.chassis.send_control_vel([i/t for i in cmd])
            time.sleep(0.02)
    
    def optimize_rel_pathxyY_control(self, path):
        ''' 
        :param path:  [[x,y,yaw],...]
        '''
        self.chassis.optimize_pathxyY_control(path)

    def cam1_data(self):
       return self.cam.get_cam1_data()
    
    def cam2_data(self):
        return self.cam.get_cam2_data()

    def cam3_data(self):
        return self.cam.get_cam3_data()
    
    # lambdaJPEG
    def cam1_compress(self):
        return self.cam.compress_image(self.cam.get_cam1_data())
    
    def cam2_compress(self):
        return self.cam.compress_image(self.cam.get_cam2_data())
    
    def cam3_compress(self):
        return self.cam.compress_image(self.cam.get_cam3_data())


import signal
def handle_sigint(signum, frame):
    # logger.info("Received SIGINT, shutting down gracefully...")
    rospy.signal_shutdown("SIGINT received")
    exit(0)

def robot_controller_access(registry_node :bool = True):
    # ubuntu 
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, handle_sigint)
    import os
    robot_type = os.environ.get('ROBOT_TYPE', 'TURTLE2')
    if robot_type == 'TURTLE2':
        return Turtle2Controller(registry_node)
    elif robot_type == 'MOVING':
        return MovingController(registry_node)
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")

# 
if __name__ == "__main__":
    turtle2 = Turtle2Controller(True)
    # 
    # turtle2.head_control([-1.0,0.0])
    # 
    # turtle2.lift_control(0.4)
    # 
    # head_data = turtle2.head_data()
    # print("head data: ", head_data)
    # 
    # lift_data = turtle2.lift_data()
    # print("lift data: ", lift_data)
    
    # 
    # global_pose = turtle2.chassis_pose_data()
    # print("global pose: ", global_pose)

    # turtle2.chassis_set_current_pose_as_virtual_zero()
    # rel_pose = turtle2.chassis_rel_pose_data()
    # print("relative pose: ", rel_pose)
    
    # # 
    # turtle2.chassis_control_relative_pose([0.0, 0.0, 1.57])

    # while True:
    #     time.sleep(1)
    # pos1 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    # pos2 = [0.0,0.0,0.05,0.0,0.0,0.0,0.0]
    # turtle2.arms_control(pos1,pos2)
    turtle2.chassis_set_current_pose_as_virtual_zero()
    turtle2.chassis_control_relative_pose([0.5, 0.0, 0.0],True)
    # turtle2.chassis_control_relative_pose([0.5, 0.0, 0.0],False)

    # turtle2.chassis_control_delta_pose([-0.3, 0.0, 0.0],True)

    for i in range(3): time.sleep(1)

