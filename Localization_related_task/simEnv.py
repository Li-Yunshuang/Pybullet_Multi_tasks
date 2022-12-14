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
    def __init__(self, bullet_client, path):
        """
        path: 模型路径
        """
        self.p = bullet_client
        self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        self.p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0, 0, 0])
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 添加路径
        self.p.loadURDF("plane.urdf", [0, 0, 0])  # 加载地面
        #self.p.loadURDF('myModel/tray/tray.urdf', [0, 0, 0])   # 加载托盘
        self.p.setGravity(0, 0, -9.8) # 设置重力

        self.flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.p.setPhysicsEngineParameter(solverResidualThreshold=0)




        # basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
        # # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
        # matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)
        # tx_vec = np.array([matrix[0], matrix[3], matrix[6]])              # 变换后的x轴
        # tz_vec = np.array([matrix[2], matrix[5], matrix[8]])              # 变换后的z轴

        # basePos = np.array(basePos)
        # # 摄像头的位置
        # # BASE_RADIUS 为 0.5，是机器人底盘的半径。BASE_THICKNESS 为 0.2 是机器人底盘的厚度。
        # # 别问我为啥不写成全局参数，因为我忘了我当时为什么这么写的。
        # cameraPos = basePos + BASE_RADIUS * tx_vec + 0.5 * BASE_THICKNESS * tz_vec
        # targetPos = cameraPos + 1 * tx_vec
    
        # viewMatrix = p.computeViewMatrix(
        # cameraEyePosition=cameraPos,
        # cameraTargetPosition=targetPos,
        # cameraUpVector=tz_vec,
        # physicsClientId=physicsClientId
        # )
        # projectionMatrix = p.computeProjectionMatrixFOV(
        # fov=50.0,               # 摄像头的视线夹角
        # aspect=1.0,
        # nearVal=0.01,            # 摄像头焦距下限
        # farVal=20,               # 摄像头能看上限
        # physicsClientId=physicsClientId
        # )


        # 加载相机
        # self.viewMatrix = self.p.computeViewMatrix([0, 0.5, 0.4], [0, 0, 0.4], [0, 0, 1])   #TYPE1
        self.viewMatrix = self.p.computeViewMatrix([-0.6, 0.6, 0.5], [0,-0.15, 0.4], [1, -1, 1])   #TYPE2 [-0.4, 0.4, 0.5], [0,-0.4,0.3], [1, -1, 1]
        self.projectionMatrix = self.p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

        # 获取urdf物体列表
        if isinstance(path, str):
            self.urdfs_list = glob.glob(os.path.join(path, '*.urdf'))
            self.urdfs_list.sort()
        elif isinstance(path, list):
            self.urdfs_list = []
            for pth in path:
                self.urdfs_list.extend(glob.glob(os.path.join(pth, '*.urdf')))
            random.shuffle(self.urdfs_list)  # 随机排序
            
        self.num_urdf = 0
        self.urdfs_id = []

        self.EulerRPList = [[0, 0], [math.pi/2, 0], [-1*math.pi/2, 0], [math.pi, 0], [0, math.pi/2], [0, -1*math.pi/2]]

        # 获取obj物体列表
        # self.objs_list = glob.glob(os.path.join(path, '*.obj'))
        # self.objs_list.sort()
        # self.num_obj = 0
        # self.objs_id = []
    
    def _urdf_nums(self):
        return len(self.urdfs_list)
    

    def init_single_mesh(self, urdfname, quaternion):
        """
        初始化mesh
        """
        # 获取obj当前位姿
        offset = [0, 0, 0]
        # quaternion = [0, 0, 0, 1]

        # 计算从obj坐标系到URDF坐标系的变换矩阵
        # 平移：self.xyz [-0.019, 0.019, -0.019]  旋转: 欧拉角[1.570796, 0, 0]
        # (1) 欧拉角->四元数
        orn = self.p.getQuaternionFromEuler([1.570796, 0, 0])
        # (2) 四元数->旋转矩阵
        rot = tool.quaternion_to_rotation_matrix(orn)
        # (3) 计算变换矩阵
        urdf_xyz = get_urdf_xyz(urdfname)
        mat = tool.getTransfMat(urdf_xyz, rot)

        # 获取obj文件路径
        objURDF_name = urdfname.replace('.urdf', '.obj')      # 单物体时使用

        # 读取obj文件，并根据scale缩放
        urdf_scale = get_urdf_scale(urdfname)
        mesh = Mesh(objURDF_name, urdf_scale)

        # 计算物体的变换矩阵(从URDF坐标系到物体坐标系)
        rotate_mat = tool.quaternion_to_rotation_matrix(quaternion)  # 四元数转旋转矩阵
        transMat = tool.getTransfMat(offset, rotate_mat)

        transMat = np.matmul(transMat, mat) # !!!! 注意乘的顺序, 使用

        # 根据旋转矩阵调整mesh顶点坐标
        mesh.transform(transMat)
        
        return mesh


    # 加载单物体
    def loadObjInURDF(self, idx, num, render_n):
        """
        以URDF的格式加载单个obj物体

        idx: 物体id
        render_n: 当前物体的渲染次数，根据此获取物体的朝向
        """
        # 获取物体文件
        self.urdfs_filename = [self.urdfs_list[idx]]
        self.num_urdf = 1
        
        print('urdf filename = ', os.path.basename(self.urdfs_filename[0]))

        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []

        # 随机方向
        # baseEuler = [self.EulerRPList[render_n][0], self.EulerRPList[render_n][1], random.uniform(0, 2*math.pi)]
        baseEuler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
        baseOrientation = self.p.getQuaternionFromEuler(baseEuler)
        # baseOrientation = [0, 0, 0, 1]    # 固定方向

        # 初始化mesh
        mesh = self.init_single_mesh(self.urdfs_filename[0], baseOrientation)
        min_z = mesh.min_z()

        # 随机位置
        pos = 0.1
        # basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), random.uniform(0.1, 0.4)] 
        basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), -1*min_z] 
        # basePosition = [0.05, 0.05, 0.1] # 固定位置

        # 加载物体
        urdf_id = self.p.loadURDF(self.urdfs_filename[0], basePosition, baseOrientation)    

        # 获取xyz和scale信息
        inf = self.p.getVisualShapeData(urdf_id)[0]

        self.urdfs_id.append(urdf_id)
        self.urdfs_xyz.append(inf[5]) 
        self.urdfs_scale.append(inf[3][0]) 

        return self.urdfs_id
    
    """
    原始加载函数
    """
    def loadObjsInURDF(self, idx, num):
        """
        以URDF的格式加载多个obj物体

        num: 加载物体的个数
        idx: 开始的id
            idx为负数时，随机加载num个物体
            idx为非负数时，从id开始加载num个物体
        """
        assert idx >= 0
        self.num_urdf = num

        # 获取物体文件
        if (idx + self.num_urdf) >= (len(self.urdfs_list) - 1):     # 这段代码主要针对加载多物体的情况
            self.num_urdf = len(self.urdfs_list) - 1 - idx
            assert self.num_urdf >= 0
        if self.num_urdf == 0:
            self.urdfs_filename = [self.urdfs_list[idx]]
            self.num_urdf = 1
        else:
            self.urdfs_filename = self.urdfs_list[idx:idx+self.num_urdf]
        
        print('self.urdfs_filename = ', self.urdfs_filename)

        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []
        for i in range(self.num_urdf):
            # 随机位置
            pos = 0.05
            basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), random.uniform(0.1, 0.4)] 
            # basePosition = [0, 0, 0.1] # 固定位置

            # 随机方向
            baseEuler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
            baseOrientation = self.p.getQuaternionFromEuler(baseEuler)
            # baseOrientation = [0, 0, 0, 1]    # 固定方向
            
            # 加载物体
            urdf_id = self.p.loadURDF(self.urdfs_filename[i], basePosition, baseOrientation)    

            # 获取xyz和scale信息
            inf = self.p.getVisualShapeData(urdf_id)[0]

            self.urdfs_id.append(urdf_id)
            self.urdfs_xyz.append(inf[5]) 
            self.urdfs_scale.append(inf[3][0]) 
    

    def removeObjsInURDF(self):
        """
        移除objs
        """
        for i in range(self.num_urdf):
            self.p.removeBody(self.urdfs_id[i])
    

    def renderURDFImage(self, save_path, num, all_num):
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
        # scio.savemat(save_path + '/camera_rgb.mat', {'A':im_rgb})
        # scio.savemat(save_path + '/camera_depth.mat', {'A':im_depthCamera})
        # scio.savemat(save_path + '/camera_depth_rev.mat', {'A':im_depthCamera_rev})
        # scio.savemat(save_path + '/camera_mask.mat', {'A':im_mask})
        print(save_path + '/'+ num +'/'+str(all_num)+'.png')
        cv2.imwrite(save_path + '/'+ num +'/'+str(all_num)+'.png', im_rgb)
        # cv2.imwrite(save_path + '/camera_mask.png', im_mask*20)
        # cv2.imwrite(save_path + '/camera_depth.png', tool.depth2Gray(im_depthCamera))
        # cv2.imwrite(save_path + '/camera_depth_rev.png', tool.depth2Gray(im_depthCamera_rev))

        print('>> 渲染结束')



