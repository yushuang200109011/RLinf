#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sensor_msgs.msg import Image
import rospy
import cv2
from cv_bridge import CvBridge
# msgs

class SensorBase:
    ''' 传感器抽象类，用于提供统一的接口和说明
    '''
    def __init__(self):
        pass
    
class Camera(SensorBase):
    def __init__(self):
        super().__init__()
        self.image1Data = None
        self.image2Data = None
        self.image3Data = None
        self.bridge = CvBridge()
        self.camera1_sub = rospy.Subscriber('/camera1/usb_cam1/image_raw', Image, self.camera1_callback)
        self.camera2_sub = rospy.Subscriber('/camera2/usb_cam2/image_raw', Image, self.camera2_callback)
        self.camera3_sub = rospy.Subscriber('/camera3/usb_cam3/image_raw', Image, self.camera3_callback)

    def compress_image(self,image_np):
        """压缩图像为 JPEG 字节流（基类通用方法）"""
        success, encoded = cv2.imencode(".jpg", image_np)
        if not success:
            raise ValueError("图像压缩失败")
        return encoded.tobytes()

    def camera1_callback(self, data):
        if data is not None:
            self.image1Data = data

    def camera2_callback(self, data):
        if data is not None:
            self.image2Data = data
        
    def camera3_callback(self, data):
        if data is not None:
            self.image3Data = data

    def get_cam1_data(self):
        cv_image = self.bridge.imgmsg_to_cv2(self.image1Data, "rgb8")
        return cv_image

    def get_cam2_data(self):
        cv_image = self.bridge.imgmsg_to_cv2(self.image2Data, "rgb8")
        return cv_image
    
    def get_cam3_data(self):
        cv_image = self.bridge.imgmsg_to_cv2(self.image3Data, "rgb8")
        return cv_image
    



