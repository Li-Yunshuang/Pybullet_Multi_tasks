"""
Objectfolder Real-time Rendering when Grasping
"""

from telnetlib import STATUS
from traceback import format_exception_only
from turtle import shearfactor
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
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRASP_GAP = 0.005
GRASP_DEPTH = 0.008

import matplotlib.pyplot as plt

#from playsound import playsound

def run():
    database_path = 'egad_eval_set/'      # 数据库路径

    cid = p.connect(p.GUI)  # 连接服务器
    env = SimEnv(p, database_path) # 初始化虚拟环境
    # Initialize Panda
    panda = panda_sim.PandaSimAuto(p, [0, -0.5, 0])     #[0, -0.5, 0]
    pandaID = panda.panda

    GRASP_STATE = False
    grasp_config = {'x':0, 'y':0, 'z':0.01, 'angle':0, 'width':0.16}
    # the unit of x y z is meter, angle is rad
    img_path = './results/Image/'

    all_num = 0    # 预设抓取次数
    obj_nums = 1    # 每次加载的物体个数
    start_idx = 0   # 开始加载的物体id

    idx = start_idx

    cnt = 0
    traj_times = 500

    while True:
        # 加载物体
        # env.loadObjsInURDF(idx, obj_nums)
        x = random.uniform(-0.2,0.2)
        y = random.uniform(-0.2,0.2)
        cup = p.loadURDF("egad_eval_set/model.urdf", [x, y, 0])    # Random appear in the tray

        idx += obj_nums
        timestep = 0
        action_list = []
        endpose_list = []
        flag = 0

        plt.ion()
        plt.figure(1)

        first_sound = 0
        while True:
            p.stepSimulation()
            p.setGravity(0,0,-9.8)
            time.sleep(1./240.)
            timestep = timestep + 1

            pos, orn = p.getBasePositionAndOrientation(cup)

            if all_num < traj_times:
                GRASP_STATE = True
            else:
                GRASP_STATE = False

            if GRASP_STATE:
                # Panda Graspinng
                status, actions = panda.step_grasping([pos[0], pos[1], pos[2]+0.03], grasp_config['angle'], (grasp_config['width'])/2)
                if actions != None:
                    last_actions = actions
                if status: #WIDTH
                    cnt = 0

                    file_path_ac = './results/actions/'+str(all_num)+'.npy'
                    file_path_end = './results/endpose/'+str(all_num)+'.npy'
                    np.save(file_path_ac, action_list)
                    np.save(file_path_end, endpose_list)
                    all_num += 1
                    first_sound = 0
                    print('>> Finish Grasping: ', all_num)

                    p.removeBody(cup)
                    break

            pts =  p.getContactPoints(bodyA = pandaID, linkIndexA = 9)   ## 9 &10  left finger contact
            pts1 =  p.getContactPoints(bodyA = pandaID, linkIndexA = 10)   ## 9 &10  Right finger contact

            if len(p.getContactPoints(bodyA = pandaID, linkIndexA = 9)) > 0:   ## 9 &10  left finger

                pts =  p.getContactPoints(bodyA = pandaID, linkIndexA = 9)   ## 9 &10  left finger
                
                c_x = 0
                c_y = 0
                c_z = 0
                shear_forces = [0,0,0]
                normal_forces = 0

                for pt in pts:
                    #print(pt)
                # ignore contacts we don't care (those not in self.objects)
                    c_x = c_x+pt[6][0]
                    c_y = c_y+pt[6][1]
                    c_z = c_z+pt[6][2]
                # Accumulate normal forces
                    normal_forces += pt[9]
                    shear_forces[0] += pt[11][0] * pt[10] + pt[13][0] * pt[12]
                    shear_forces[1] += pt[11][1] * pt[10] + pt[13][1] * pt[12]
                    shear_forces[2] += pt[11][2] * pt[10] + pt[13][2] * pt[12]
                print(normal_forces)

                contacts = p.getContactPoints(pandaID, cup)

                pos, orn = p.getBasePositionAndOrientation(cup)          # get  cup T  world
                inv_cup_pos, inv_cup_orn = p.invertTransform(pos, orn)   # get  world T cup
                obj_pos_in_cup, obj_orn_in_cup = p.multiplyTransforms([c_x/len(pts),c_y/len(pts),c_z/len(pts)],p.getQuaternionFromEuler((0,0,0)),inv_cup_pos, inv_cup_orn)
                
                # Get force directly from pb.functions

                #force_pos_in_cup, force_orn_in_cup = p.multiplyTransforms((0, 0, 0),p.getQuaternionFromEuler(contacts[0][7]),inv_cup_pos, inv_cup_orn)
                #point T cup = point T world * world T cup
                # force_vector = contacts[0][7]   #contactNormalOnB：contact vector，from B to A，vec3, list of 3 floats 
                # scale = contacts[0][9]   #normalForce： normal force in last stepSimulation
                # forces = np.array([[scale*force_vector[0], scale*force_vector[1], scale*force_vector[2]]])

                # print("-----------!!!CONTACT!!!-----------")
                # print("Current obj pos: "+str(pos))
                # #print(contacts[0][5])   # Points on Panda
                # print("Contact force vector: "+str(forces))
                # print("Current obj contact pos in WORLD base: "+str(contacts[0][6]))  # Points on cup
                # print("Current obj contact pos in OBJ base: "+str(obj_pos_in_cup))
                # print("Current obj contact orn in OBJ base: "+str(obj_orn_in_cup))

                
                print("****************CONTACT*******************")
                # print("Current endpose"+str(panda_pose))
                file_path = './results/touch/'+str(all_num)+'/'

                # Show tactile image in real-time
                if timestep%4 == 0:
                    path = TouchNet_eval(obj_pos_in_cup, cnt, file_path, normal_forces)
                    plt.cla()
                    img = Image.open(path)
                    img = img.transpose(Image.ROTATE_270)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.show()
                    plt.pause(0.01)
                    env.renderURDFImage(img_path, all_num, str(cnt))
                    
                # Generate SOund

                # if first_sound == 0:
                #     # gensound 
                #     ps = AudioNet_eval(obj_pos_in_cup, normal_forces, cnt)
                #     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                #     os.system('python /home/lys/A_grasping_panda/results/audio/test.py')
                #     # play sound 
                #     #playsound(ps)
                #     first_sound = 1


                #AudioNet_eval(obj_pos_in_cup, forces, cnt)
                cnt = cnt +1

            panda_pose,panda_orn = p.getLinkState(pandaID,11)[:2]


            # Save both pose and actions for the sequences
            # if flag:
            #     action_new = [panda_pose[0]-last_pose[0], panda_pose[1]-last_pose[1], panda_pose[2]-last_pose[2]]
            # file_path = './results/touch/'+str(all_num)+'/'
            #     #gelinfo = [0. 0. 0.001]
            # if timestep%8 == 0:
            #         #TouchNet_eval(obj_pos_in_cup, cnt, file_path)
            #         #if actions == None:
            #         #    print(last_actions)
            #         #    action_list.append(last_actions)
            #         #else:
            #         #    print(actions)
            #         #    action_list.append(actions)
            #     if timestep != 0:
            #         action_list.append(action_new)
            #     print(panda_pose)
            #     endpose_list.append(panda_pose)
            #         #AudioNet_eval(obj_pos_in_cup, forces, cnt)
            #     env.renderURDFImage(img_path, all_num, str(cnt))
            #     cnt = cnt +1

            pos, orn = p.getBasePositionAndOrientation(cup)
            # print(pos)


 

if __name__ == "__main__":
    run()
