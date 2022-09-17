"""
Pointing to certain points along normal vector direction:

In this file, panda robot point to certion vertex on the object along the normal vector.

Args:
--num: the serial number of object
--start: the serial number of data in the npy files

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



def run(num, start):
    database_path = 'egad_eval_set/'      # 数据库路径
    cid = p.connect(p.GUI)  # 连接服务器
    env = SimEnv(p, database_path) # 初始化虚拟环境

    # 初始化panda机器人
    panda = panda_sim.PandaSimAuto(p, [0, -0.5, 0])     #(p, pos, ori)
    pandaID = panda.panda

    GRASP_STATE = False
    grasp_config = {'x':0, 'y':0, 'z':0.01, 'angle':0, 'width':0.16}

    img_path = 'img/img_urdf'   # IMage save directory 

    all_num = 0     # 预设抓取次数
    count = start
    data = np.load('./contacts_new/'+num+'/'+str(count)+'.npy')   # Load npy files
    
    while True:
        cup = p.loadURDF("ObjectFolder201-300/"+num+"/model.urdf",[0,-0.15,0.4],p.getQuaternionFromEuler([0, 0, 0]),useFixedBase=True)  #[0.5 0.5 0]

        while True:


            p.stepSimulation()
            p.setGravity(0,0,-9.8)
            time.sleep(1./240.)

            data = np.load('./contacts_new/'+num+'/'+str(count)+'.npy')

            traj = [data[0], data[1], data[2]]                   # Position of Vertex
            len = np.sqrt(data[3] ** 2+data[4]**2+data[5]**2)    # Length of normal vector
            gripper = [data[3]/len, data[4]/len, data[5]/len]    # Normalize normal vector

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


            delta = [0, 0, -0.05]                             # Offset for camera along normal vector
            del_x = np.sin(pitch)*np.cos(yaw) * delta[2]
            del_y = -np.sin(yaw) * delta[2]
            del_z = np.cos(pitch)*np.cos(yaw) * delta[2]

            pos, orn = p.getBasePositionAndOrientation(cup)          # get  cup T  world
            obj_pos_in_W, obj_orn_in_W = p.multiplyTransforms(traj,p.getQuaternionFromEuler((0,0,0)),pos, orn)


            keys = p.getKeyboardEvents()

            if count < start + 10:
                GRASP_STATE = True
            else:
                p.disconnect(p.GUI)
            if GRASP_STATE:
                   if panda.localize([obj_pos_in_W[0]+del_x, obj_pos_in_W[1]+del_y, obj_pos_in_W[2]+del_z], [roll, pitch, yaw], (grasp_config['width'])/2): #WIDTH
                    cnt = 0
                    GRASP_STATE = False
                    all_num += 1
                    env.renderURDFImage(img_path, num, count)
                    count = count +1
                    print('>> 抓取完毕: ', all_num)
                    break         
            if ord('3') in keys and keys[ord('3')]&p.KEY_WAS_TRIGGERED:
                env.removeObjsInURDF()
                break

            pos, orn = p.getBasePositionAndOrientation(cup)

    

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--num', type=str, default = None)
    parser.add_argument('--start', type=int, default = None)
    
    args = parser.parse_args()
    run(args.num, args.start)
