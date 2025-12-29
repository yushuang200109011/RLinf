import numpy as np
import rospy
from turtle2_controller.kinematics import kinematics
from turtle2_controller.controllers import HeadController,ChassisController,LiftController,ArmsController,MovingHeadController,MovingChassisController,MovingLiftController
from turtle2_controller.sensors import Camera
import time
'''
本文件用于抽象乌龟的所有IO接口,如头部的控制接口，数据接收接口等。
'''

class RobotController:
    ''' 控制器基类，用于提供统一的接口和说明
    '''
    def __init__(self):
        pass

    def head_control(self, cmd):
        ''' 发送头部控制指令
        :param cmd : 头部控制指令
        '''
        pass
    def head_data(self):
        ''' 获取头部数据
        :return: [pitch,yaw]
        '''
        pass
    def lift_control(self, cmd):
        ''' 发送升降机控制指令
        :param lift_cmd: 升降机控制指令
        '''
        pass
    def lift_data(self):
        ''' 获取升降机数据
        :return: 升降机数据
        '''
        pass
    def chassis_control(self, cmd):
        ''' 发送底盘控制指令
        :param chassis_cmd: 底盘控制指令
        '''
        pass
    def chassis_data(self):
        ''' 获取底盘数据
        :return: 底盘数据
        '''
        pass
    def get_infer_flag(self):
        infer_flag=rospy.get_param("/master_teach_mode",0)
        return (infer_flag==0)

 
class Turtle2Controller(RobotController):
    ''' 乌龟2号整机控制的实现
    用法： 实例化该类，如 c = Turtle2Controller()
    某个组件的读取 c.xxx_data() # 返回数据形式不一，需要详细参考各个组件的控制类
    某个组件的控制 c.xxx_control(cmd) # cmd形式不一,需要详细参考各个组件的控制类
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
        rospy.Rate(2).sleep() # 等待节点启动完成

        self.virtual_zero_tf_inv = None

    def head_control(self, cmd):
        ''' 发送头部控制指令
        :param cmd : 头部控制指令
        '''
        self.head.send_control(cmd)

    def head_data(self):
        ''' 获取头部数据
        :return: list([pitch,yaw])
        '''
        return self.head.get_data()

    def lift_control(self,cmd):
        ''' 发送升降机控制指令
        :param lift_cmd: 升降机控制指令
        '''
        self.lift.send_control(cmd)

    def lift_data(self):
        ''' 获取升降机数据
        :return: 升降机数据
        '''
        return self.lift.get_data()
    
    def arms_start(self):
        """ 插值模式下开启机械臂控制
        """
        self.arms.start()


    def arms_stop(self):
        """ 插值模式下关闭机械臂控制
        """
        self.arms.stop()

    
    def arms_control(self,cmd_l,cmd_r):
        ''' 发送机械臂控制指令
        :param arm_l&arm_r: 机械臂控制指令 [x,y,z,roll,pitch,yaw,gripper]
        '''
        self.arms.send_control(cmd_l,cmd_r)

    def arms_control_pose_trj(self, is_async_: bool, pose_trj_l, pose_trj_r, pos_method='linear', quaternion_interpolation_method="slerp" ,infer_time=0.05, step_time=0.005, interpolation_step=200):
        ''' 发送机械臂位置轨迹控制指令
        :param pose_trj_l&pose_trj_r: 机械臂位置轨迹控制指令 [[x,y,z,roll,pitch,yaw,gripper],...]
        :method : 轨迹插值方法,可选'slerp'或'toppra'
        '''
        self.arms.send_control_pose_trj(is_async_, pose_trj_l, pose_trj_r, pos_method, quaternion_interpolation_method, infer_time, step_time, interpolation_step)

    def arms_control_raw_trj(self,cmd_l, cmd_r,t_step=0.005):
        """ 发送机械臂原始轨迹控制指令
        :param cmd_l: 左臂原始轨迹控制指令 [[x,y,z,roll,pitch,yaw,gripper],...]
        :param cmd_r: 右臂原始轨迹控制指令 [[x,y,z,roll,pitch,yaw,gripper],...]
        :param t_step: 轨迹控制时间步长
        该函数用于发送机械臂的原始轨迹控制指令，适用于需要精确控制机械臂运动的场景。该函数会直接发送原始轨迹指令，不进行任何插值处理。
        """
        self.arms.send_control_raw_trj(cmd_l, cmd_r, t_step)

    def start_arm_actions_thread(self):
        self.arms.start_arm_actions_thread()

    def arms_data(self):
        ''' 获取机械臂末端数据
        :return:  arm_l[x,y,z,roll,pitch,yaw,gripper], arm_r[x,y,z,roll,pitch,yaw,gripper]
        '''
        return self.arms.get_data() # left_arm, right_arm
    
    def arms_cur_data(self):
        ''' 获取机械臂关节电流'''
        return self.arms.get_cur_data()
    
    def arms_joint_data(self):
        ''' 获取机械臂关节角度
        :return: arm_l[joint1,joint2,joint3,joint4,joint5,joint6], arm_r[joint1,joint2,joint3,joint4,joint5,joint6]
        '''
        return self.arms.get_pos_data()

    def arms_zero(self):
        ''' 双臂直接归零
        '''
        self.arms.send_zero()
    
    def arms_zero_after_oepen_gripper(self):
        ''' 双臂先张开夹爪再归零
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
        ''' 发送底盘速度控制指令
        :param cmd: 底盘控制指令 [vx,vy,dyaw]
        '''
        self.chassis.send_control_vel(cmd)

    def chassis_pose_data(self):
        ''' 获取底盘数据
        :return: 底盘数据 [x,y,yaw]
        '''
        return self.chassis.pose_data()
    
    def chassis_rel_pose_data(self):
        ''' 获取底盘数据
        :return: 底盘数据 [x,y,yaw]
        '''
        return self.chassis.rel_pose_data()
    
    def chassis_control_global_pose(self,cmd):
        ''' 发送底盘全局位置控制
        :param cmd: 底盘控制指令 [x,y,yaw]
        '''
        self.chassis.send_control_global_pose(cmd)

    def chassis_set_virtual_zero(self,pose):
        ''' 设置底盘定位虚拟零点,相对位置控制函数会以这个为基准
        :param pose: 虚拟零点坐标 [x,y,yaw]
        @todo :暂不开放
        '''
        self.chassis.set_virtual_zero(pose)

    def chassis_set_current_pose_as_virtual_zero(self):
        ''' 设置当前底盘位置为虚拟零点,相对位置控制函数会以这个为基准
        '''
        self.chassis.set_current_pose_as_virtual_zero()

    def chassis_stop(self):
        """ 停止底盘运动
        该函数会停止底盘运动，并清空目标队列
        """
        self.chassis.stop()

    def chassis_start(self):
        """ 开始底盘运动
        该函数会开始底盘运动，并清空目标队列
        """
        self.chassis.start()

    def chassis_control_relative_pose(self, cmd, is_arrived=False):
        ''' 发送底盘相对位置控制指令
        @param cmd: 底盘控制指令 [x,y,yaw]
        @param is_arrived: 是否到达目标位置的标志位，默认为False
        '''
        return self.chassis.send_control_relative_pose(cmd, is_arrived)

    def chassis_control_delta_pose(self,cmd,is_arrived=False):
        ''' 发送底盘增量位置控制指令
        :param cmd: 底盘控制指令 [x,y,yaw]
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
        ''' 优化路径控制
        :param path: 需要优化的路径 [[x,y,yaw],...]
        '''
        self.chassis.optimize_pathxyY_control(path)

    def cam1_data(self):
       return self.cam.get_cam1_data()
    
    def cam2_data(self):
        return self.cam.get_cam2_data()

    def cam3_data(self):
        return self.cam.get_cam3_data()
    
    # 使用lambda表达式压缩相机图像为JPEG字节流
    def cam1_compress(self):
        return self.cam.compress_image(self.cam.get_cam1_data())
    
    def cam2_compress(self):
        return self.cam.compress_image(self.cam.get_cam2_data())
    
    def cam3_compress(self):
        return self.cam.compress_image(self.cam.get_cam3_data())


class MovingController(RobotController):
    ''' 乌龟2号整机控制的实现
    用法： 实例化该类，如 c = MovingController()
    某个组件的读取 c.xxx_data() # 返回数据形式不一，需要详细参考各个组件的控制类
    某个组件的控制 c.xxx_control(cmd) # cmd形式不一,需要详细参考各个组件的控制类
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
        rospy.Rate(2).sleep() # 等待节点启动完成

        self.virtual_zero_tf_inv = None

    def head_control(self, cmd):
        ''' 发送头部控制指令
        :param cmd : 头部控制指令
        '''
        self.head.send_control(cmd)

    def head_data(self):
        ''' 获取头部数据
        :return: list([pitch,yaw])
        '''
        return self.head.get_data()

    def lift_control(self,cmd):
        ''' 发送升降机控制指令
        :param lift_cmd: 升降机控制指令
        '''
        self.lift.send_control(cmd)

    def lift_data(self):
        ''' 获取升降机数据
        :return: 升降机数据
        '''
        return self.lift.get_data()
    
    def arms_start(self):
        """ 插值模式下开启机械臂控制
        """
        self.arms.start()


    def arms_stop(self):
        """ 插值模式下关闭机械臂控制
        """
        self.arms.stop()

    def arms_control(self,cmd_l,cmd_r):
        ''' 发送机械臂控制指令
        :param arm_l&arm_r: 机械臂控制指令 [x,y,z,roll,pitch,yaw,gripper]
        '''
        self.arms.send_control(cmd_l,cmd_r)

    def arms_control_pose_trj(self, is_async_: bool, pose_trj_l, pose_trj_r, pos_method='linear', quaternion_interpolation_method="slerp" ,infer_time=0.05, step_time=0.005, interpolation_step=200):
        ''' 发送机械臂位置轨迹控制指令
        :param pose_trj_l&pose_trj_r: 机械臂位置轨迹控制指令 [[x,y,z,roll,pitch,yaw,gripper],...]
        :method : 轨迹插值方法,可选'slerp'或'toppra'
        '''
        self.arms.send_control_pose_trj(is_async_, pose_trj_l, pose_trj_r, pos_method, quaternion_interpolation_method, infer_time, step_time, interpolation_step)

    def arms_control_raw_trj(self,cmd_l, cmd_r,t_step=0.005):
        """ 发送机械臂原始轨迹控制指令
        :param cmd_l: 左臂原始轨迹控制指令 [[x,y,z,roll,pitch,yaw,gripper],...]
        :param cmd_r: 右臂原始轨迹控制指令 [[x,y,z,roll,pitch,yaw,gripper],...]
        :param t_step: 轨迹控制时间步长
        该函数用于发送机械臂的原始轨迹控制指令，适用于需要精确控制机械臂运动的场景。该函数会直接发送原始轨迹指令，不进行任何插值处理。
        """
        self.arms.send_control_raw_trj(cmd_l, cmd_r, t_step)
    def start_arm_actions_thread(self):
        self.arms.start_arm_actions_thread()

    def arms_data(self):
        ''' 获取机械臂末端数据
        :return:  arm_l[x,y,z,roll,pitch,yaw,gripper], arm_r[x,y,z,roll,pitch,yaw,gripper]
        '''
        return self.arms.get_data() # left_arm, right_arm
    
    def arms_cur_data(self):
        ''' 获取机械臂关节电流'''
        return self.arms.get_cur_data()
    
    def arms_joint_data(self):
        ''' 获取机械臂关节角度
        :return: arm_l[joint1,joint2,joint3,joint4,joint5,joint6], arm_r[joint1,joint2,joint3,joint4,joint5,joint6]
        '''
        return self.arms.get_pos_data()

    def arms_zero(self):
        ''' 双臂直接归零
        '''
        self.arms.send_zero()
    
    def arms_zero_after_oepen_gripper(self):
        ''' 双臂先张开夹爪再归零
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
        ''' 发送底盘速度控制指令
        :param cmd: 底盘控制指令 [vx,vy,dyaw]
        '''
        self.chassis.send_control_vel(cmd)

    def chassis_pose_data(self):
        ''' 获取底盘数据
        :return: 底盘数据 [x,y,yaw]
        '''
        return self.chassis.pose_data()
    
    def chassis_rel_pose_data(self):
        ''' 获取底盘数据
        :return: 底盘数据 [x,y,yaw]
        '''
        return self.chassis.rel_pose_data()
    
    def chassis_control_global_pose(self,cmd):
        ''' 发送底盘全局位置控制
        :param cmd: 底盘控制指令 [x,y,yaw]
        '''
        self.chassis.send_control_global_pose(cmd)

    def chassis_set_virtual_zero(self,pose):
        ''' 设置底盘定位虚拟零点,相对位置控制函数会以这个为基准
        :param pose: 虚拟零点坐标 [x,y,yaw]
        @todo :暂不开放
        '''
        self.chassis.set_virtual_zero(pose)

    def chassis_set_current_pose_as_virtual_zero(self):
        ''' 设置当前底盘位置为虚拟零点,相对位置控制函数会以这个为基准
        '''
        self.chassis.set_current_pose_as_virtual_zero()

    def chassis_stop(self):
        """ 停止底盘运动
        该函数会停止底盘运动，并清空目标队列
        """
        self.chassis.stop()

    def chassis_start(self):
        """ 开始底盘运动
        该函数会开始底盘运动，并清空目标队列
        """
        self.chassis.start()

    def chassis_control_relative_pose(self,cmd,is_arrived=False):
        ''' 发送底盘相对位置控制指令
        :param cmd: 底盘控制指令 [x,y,yaw]
        '''
        return self.chassis.send_control_relative_pose(cmd,is_arrived)

    def chassis_control_delta_pose(self,cmd):
        ''' 发送底盘增量位置控制指令
        :param cmd: 底盘控制指令 [x,y,yaw]
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
        ''' 优化路径控制
        :param path: 需要优化的路径 [[x,y,yaw],...]
        '''
        self.chassis.optimize_pathxyY_control(path)

    def cam1_data(self):
       return self.cam.get_cam1_data()
    
    def cam2_data(self):
        return self.cam.get_cam2_data()

    def cam3_data(self):
        return self.cam.get_cam3_data()
    
    # 使用lambda表达式压缩相机图像为JPEG字节流
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
    # 获取ubuntu 环境变量
    signal.signal(signal.SIGINT, handle_sigint)
    import os
    robot_type = os.environ.get('ROBOT_TYPE', 'TURTLE2')
    if robot_type == 'TURTLE2':
        return Turtle2Controller(registry_node)
    elif robot_type == 'MOVING':
        return MovingController(registry_node)
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")

# 测试代码
if __name__ == "__main__":
    turtle2 = Turtle2Controller(True)
    # 测试头部控制
    # turtle2.head_control([-1.0,0.0])
    # 测试升降机控制
    # turtle2.lift_control(0.4)
    # 测试头部数据
    # head_data = turtle2.head_data()
    # print("head data: ", head_data)
    # 测试升降机数据
    # lift_data = turtle2.lift_data()
    # print("lift data: ", lift_data)
    
    # 测试底盘控制
    # global_pose = turtle2.chassis_pose_data()
    # print("global pose: ", global_pose)

    # turtle2.chassis_set_current_pose_as_virtual_zero()
    # rel_pose = turtle2.chassis_rel_pose_data()
    # print("relative pose: ", rel_pose)
    
    # # 发送相对位置控制指令
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

