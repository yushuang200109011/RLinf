import math
import numpy as np
import os

class kinematics:
    def __init__(self):
        # 实例化七个变换矩阵
        self.trasformation_matrix = [np.zeros((4,4)) for i in range(7)]
        pos_tf = [np.zeros(3) for i in range(7)]
        qua_tf = [np.zeros(4) for i in range(7)]
        for pos,qua,tf in zip(pos_tf,qua_tf,self.trasformation_matrix):
            pos = np.array([0,0,0])
            qua = np.array([0,0,0,1])
            self.tf = self.pose_to_transformation_matrix(pos,qua)

    def update_tf(self,position_l,quaternion_l):
        for i in range(7):
            self.trasformation_matrix = self.pose_to_transformation_matrix(position_l[i],quaternion_l[i])

    '''四元数转旋转矩阵
    '''
    def quaternion_to_rotation_matrix(self,quaternion):
        x, y, z, w = quaternion
        R = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y),0],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x),0],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2),0],
            [0,0,0,1]
        ])
        return R
    
    '''旋转，平移转变换矩阵Ｔ
    '''
    def pose_to_transformation_matrix(self,position,quaternion):
        R = self.quaternion_to_rotation_matrix(quaternion)
        x, y, z = position
        t = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]])
        T = np.dot(t,R)
        return T
    
    def pos_euler_to_transformation_matrix(self,position,euler):
        quaternion = self.euler_angle_to_quaternion(euler)
        return self.pose_to_transformation_matrix(position,quaternion)
    
    ''' 旋转矩阵转四元数
    '''
    def rotation_matrix_to_quaternion(self, R):
        # R 应该是一个 3x3 旋转矩阵
        R = R[0:3,0:3]
        q = np.zeros(4)
        K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tr = np.trace(R)

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        return [x,y,z,w]

    ''' 变换矩阵转位置和四元数
    '''
    def transformation_matrix_to_pos_qua(self,T):
        position = T[:3,3]
        R = T[:3,:3]
        quaternion = self.rotation_matrix_to_quaternion(R)
        return position, quaternion
    
    ''' 变换矩阵转位置和欧拉角
    '''
    def transformation_matrix_to_pos_euler(self,T):
        position = T[:3,3]
        R = T[:3,:3]
        quaternion = self.rotation_matrix_to_quaternion(R)
        euler = self.quaternion_to_euler_angle(quaternion)
        return position, euler

    def fk_solver(self):
        t = self.trasformation_matrix[0]
        for i in range(1,7):
            t = np.dot(t,self.trasformation_matrix[i])
        return t

    '''四元数转欧拉角
    '''
    def quaternion_to_euler_angle(self,quaternion):
        x, y, z, w = quaternion
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.atan2(t3, t4)
        return X, Y, Z
    
    '''欧拉角转四元数
    '''
    def euler_angle_to_quaternion(self, euler):
        roll, pitch, yaw = euler
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        return [x, y, z, w]
    

