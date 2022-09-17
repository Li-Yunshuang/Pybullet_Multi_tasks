"""
Directly generate images for certain vertex along normal vector
"""

from traceback import format_exception_only
import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy as np
from simEnv import SimEnv
import panda_sim_grasp as panda_sim

############OBJ Package##############
import os
import datetime
import sys
import torch
import torch.nn as nn
import numpy as np
import imageio
import json
import random
import time
from tqdm import tqdm, trange
import scipy
import librosa
from scipy.io.wavfile import write
from scipy.spatial import KDTree
import objfolder.ddsp_torch as ddsp
import itertools
from objfolder.taxim_render import TaximRender
from PIL import Image
import argparse
from objfolder.load_osf import load_osf_data

from objfolder import TouchNet_utils
from objfolder import TouchNet_model

from objfolder.utils import *
from obj import TouchNet_eval
from obj import AudioNet_eval
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRASP_GAP = 0.005
GRASP_DEPTH = 0.008

import argparse



def run(num):
    database_path = 'egad_eval_set/'      



    GRASP_STATE = False


    img_path = 'img/img_urdf'

    all_num = 0     # 预设抓取次数
    obj_nums = 1    # 每次加载的物体个数
    start_idx = 0   # 开始加载的物体id




    count = 0
    data = np.load('./contacts_new/'+num+'/'+str(count)+'.npy')
    yaw_list = []
    
    while True:
        cup = p.loadURDF("ObjectFolder/"+num+"/model.urdf",[0,0,0.5],p.getQuaternionFromEuler([0, 0, 0]),useFixedBase=True)  #[0.5 0.5 0]
        while True:


            p.stepSimulation()
            p.setGravity(0,0,-9.8)
            time.sleep(1./240.)

            if count == 100:
                file_path = './results/'+str(all_num)+'.npy'
                np.save(file_path, yaw_list)

            data = np.load('./contacts_new/'+num+'/'+str(count)+'.npy')

            traj = [data[0], data[1], data[2]]                 #vertex
            len = np.sqrt(traj[0] ** 2+traj[1]**2+traj[2]**2)
            gripper = [traj[0]/len, traj[1]/len,traj[2]/len]   #nomal vector


            pitch = np.arcsin(-gripper[2])  # arcsin(-z)
            yaw1 = np.arcsin(gripper[1]/ np.cos(pitch))        # arcsin(y/cos(pitch))
            yaw2 = np.arccos(gripper[0] / np.cos(pitch))       # arccos(x/cos(pitch))
            #print ( np.sin(yaw1)*np.cos(pitch)) #y
            if np.sin(yaw1)*np.cos(pitch) == gripper[1]:
                yaw = yaw1
                #print("1")
            else:
                yaw = yaw2

            pitch = pitch- math.pi/2
            roll = 0   

            pos, orn = p.getBasePositionAndOrientation(cup)          # get  cup T  world

            obj_pos_in_W, obj_orn_in_W = p.multiplyTransforms(traj,p.getQuaternionFromEuler((0,0,0)),pos, orn)


            if count < 100:
                delta = [0, 0, -0.05]
                del_x = np.sin(pitch)*np.cos(yaw) * delta[2]
                del_y = -np.sin(yaw) * delta[2]
                del_z = np.cos(pitch)*np.cos(yaw) * delta[2]

                cam_pos = obj_pos_in_W[0]+del_x, obj_pos_in_W[1]+del_y, obj_pos_in_W[2]+del_z

                tar_pos = obj_pos_in_W[0], obj_pos_in_W[1], obj_pos_in_W[2]

                orn = p.getQuaternionFromEuler([roll, pitch, yaw]) 
                matrix = p.getMatrixFromQuaternion(orn)
                tz_vec = np.array([matrix[2], matrix[5], matrix[8]])              # 变换后的z轴
                ty_vec = np.array([matrix[1], matrix[4], matrix[7]])               #x axis
                tx_vec = np.array([matrix[0], matrix[3], matrix[6]])  
                Z = tz_vec[0]*del_x + tz_vec[1]*del_y+tz_vec[2]*del_z
                X = tx_vec[0]*del_x + tx_vec[1]*del_y+tx_vec[2]*del_z

                print(tx_vec)

                print(X)
                print(tz_vec)
                print(Z)
                viewMatrix = p.computeViewMatrix(
                        cameraEyePosition=cam_pos,        # from cam_pos
                        cameraTargetPosition=tar_pos,     # to tar_pos
                        cameraUpVector=tz_vec,          
                    )
                projectionMatrix = p.computeProjectionMatrixFOV(
                        fov=30.0,               # 摄像头的视线夹角
                        aspect=1.0,
                        nearVal=0.001,            # 摄像头焦距下限
                        farVal=200,               # 摄像头能看上限
                    )
                img_camera = p.getCameraImage(400, 400, viewMatrix, projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                w = img_camera[0]      # width of the image, in pixels
                h = img_camera[1]      # height of the image, in pixels
                rgba = img_camera[2]    # color data RGB

                im_rgb = np.reshape(rgba, (h, w, 4))[:, :, [2, 1, 0]]
                im_rgb = im_rgb.astype(np.uint8)


                print(img_path + '/'+ num +'/'+str(count)+'.png')
                cv2.imwrite(img_path + '/'+ num +'/'+str(count)+'.png', im_rgb)                  
                    
                count = count +1
                print('>> Image: ', count)
                yaw_list.append(yaw)

                break
            else:
                p.disconnect(p.GUI)

            pos, orn = p.getBasePositionAndOrientation(cup)


 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--num', type=str, default = None)
    
    args = parser.parse_args()
    run(args.num,)
