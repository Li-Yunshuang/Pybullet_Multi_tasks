"""
虚拟环境文件
初始化虚拟环境，加载物体，渲染图像，保存图像

(待写) ！！ 保存虚拟环境状态，以便离线抓取测试
"""

import pybullet as p
import pybullet_data
import time
import math
import os
import glob
import random
import cv2
import shutil
import numpy as np
import scipy.io as scio
from mesh import Mesh
import tool

# 图像尺寸
IMAGEWIDTH = 640
IMAGEHEIGHT = 480

nearPlane = 0.01
farPlane = 10

fov = 60    # 垂直视场 图像高tan(30) * 0.7 *2 = 0.8082903m
aspect = IMAGEWIDTH / IMAGEHEIGHT

size=(0.8, 0.8)     # 桌面深度图实际尺寸 m
unit=0.0002          # 每个像素的长度 0.1mm


def get_urdf_xyz(filename):
    """
    获取urdfs_xyz
    filename: urdf文件名
    """
    with open(filename) as f:
        line = f.readlines()[15][32:-5]
        strs = line.split(" ")
        return [float(strs[0]), float(strs[1]), float(strs[2])]

def get_urdf_scale(filename):
    """
    获取urdfs_scale
    filename: urdf文件名
    """
    with open(filename) as f:
        line = f.readlines()[17]
        idx = line.find('scale') + 7
        strs = line[idx:-5].split(" ")
        return float(strs[0])



class SimEnv(object):
    """
    虚拟环境类
    """
    def __init__(self, bullet_client):
        """
        path: 模型路径
        """
        self.p = bullet_client
        self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        self.p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0, 0, 0])
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 添加路径
        self.p.loadURDF("plane.urdf", [0, 0, 0])  # 加载地面
        self.p.loadURDF('myModel/tray/tray.urdf', [0, 0, 0])   # 加载托盘
        self.p.setGravity(0, 0, -9.8) # 设置重力

        self.flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.p.setPhysicsEngineParameter(solverResidualThreshold=0)

        # 加载相机
        #self.viewMatrix = self.p.computeViewMatrix([0, 0, 0.7], [0, 0, 0], [0, 1, 0])
        self.viewMatrix = self.p.computeViewMatrix([0, 0.5, 0.4], [0, 0, 0.4], [0, 0, 1]) 
        self.projectionMatrix = self.p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        self.num_urdf = 0
        self.urdfs_id = []

        self.EulerRPList = [[0, 0], [math.pi/2, 0], [-1*math.pi/2, 0], [math.pi, 0], [0, math.pi/2], [0, -1*math.pi/2]]

    def renderURDFImage(self, save_path, index, name):
        """
        渲染图像
        """
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # ======================== 渲染相机深度图 ========================
        print('>> 渲染相机深度图...')
        # 渲染图像
        img_camera = self.p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_camera[0]      # width of the image, in pixels
        h = img_camera[1]      # height of the image, in pixels
        rgba = img_camera[2]    # color data RGB
        dep = img_camera[3]    # depth data
        mask = img_camera[4]    # mask data

        # 获取彩色图像
        im_rgb = np.reshape(rgba, (h, w, 4))[:, :, [2, 1, 0]]
        im_rgb = im_rgb.astype(np.uint8)

        # 获取深度图像
        depth = np.reshape(dep, (h, w))  # [40:440, 120:520]
        A = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane * nearPlane
        B = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane
        C = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * (farPlane - nearPlane)
        # im_depthCamera = A / (B - C * depth)  # 单位 m
        im_depthCamera = np.divide(A, (np.subtract(B, np.multiply(C, depth))))  # 单位 m
        im_depthCamera_rev = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float) * im_depthCamera.max() - im_depthCamera # 反转深度

        # 获取分割图像
        im_mask = np.reshape(mask, (h, w))

        # 保存图像
        # print('>> 保存相机深度图')
        #scio.savemat(save_path + '/camera_rgb.mat', {'A':im_rgb})
        #scio.savemat(save_path + '/camera_depth.mat', {'A':im_depthCamera})
        #scio.savemat(save_path + '/camera_depth_rev.mat', {'A':im_depthCamera_rev})
        #scio.savemat(save_path + '/camera_mask.mat', {'A':im_mask})

        cv2.imwrite(save_path + '/'+str(index) +'/'+name+'.png', im_rgb)
        # cv2.imwrite(save_path + '/camera_mask.png', im_mask*20)
        #cv2.imwrite(save_path + '/camera_depth.png', tool.depth2Gray(im_depthCamera))
        #cv2.imwrite(save_path + '/camera_depth_rev.png', tool.depth2Gray(im_depthCamera_rev))

        print('>> 渲染结束')



