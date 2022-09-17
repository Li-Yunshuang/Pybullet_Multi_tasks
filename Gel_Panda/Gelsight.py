"""
Panda equiped with Gelsight

"""

from telnetlib import STATUS
from traceback import format_exception_only
import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy as np
from simEnv import SimEnv
import panda_sim_test as panda_sim

import taxim_robot
import utils

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
import itertools
from PIL import Image
import argparse

import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRASP_GAP = 0.005
GRASP_DEPTH = 0.008


def _align_image(img1, img2):
    img_size = [480, 640]
    new_img = np.zeros([img_size[0], img_size[1] * 2, 3], dtype=np.uint8)
    new_img[:img1.shape[0], :img1.shape[1]] = img2[..., :3]
    new_img[:img2.shape[0], img_size[1]:img_size[1] + img2.shape[1], :] = (img1[..., :3])[..., ::-1]
    return new_img


def run():

    cid = p.connect(p.GUI)  # Connect to server
    env = SimEnv(p) # Initialize simulation environment
    # Initialize Panda Robot
    panda = panda_sim.PandaSimAuto(p, [0, -0.5, 0])     #[0, -0.5, 0]
    pandaID = panda.panda

    GRASP_STATE = False
    grasp_config = {'x':0, 'y':0, 'z':0.01, 'angle':0, 'width':0.16}
    # The unit of x y z width is meter, the unit of angle is rad

    gelsight = taxim_robot.Sensor(width=640, height=480)
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=15, cameraPitch=-15,
                                  cameraTargetPosition=[0.5, 0, 0.08])
    
    cam = utils.Camera(p, [640, 480])


    nbJoint = p.getNumJoints(pandaID)
    print(nbJoint)
    jointNames = {}
    for i in range(nbJoint):
        name = p.getJointInfo(pandaID, i)[1].decode()
        jointNames[name] = i

    sensorLinks =  [jointNames[name] for name in ["guide_joint_finger_left"]]    # Gelsight link
    print(sensorLinks)
    gelsight.add_camera(pandaID, sensorLinks)
    
    color, depth = gelsight.render()
    gelsight.updateGUI(color, depth)

    save_dir = os.path.join('data', 'seq')
    os.makedirs(save_dir, exist_ok=True)
    
    visualize_data = []
    tactileColor_tmp, _ = gelsight.render()
    visionColor_tmp, _ = cam.get_image()
    visualize_data.append(_align_image(tactileColor_tmp[0], visionColor_tmp))

    ## for seq data
    vision_size, tactile_size = visionColor_tmp.shape, tactileColor_tmp[0].shape
    video_path = os.path.join(save_dir, "demo.mp4")
    rec = utils.video_recorder(vision_size, tactile_size, path=video_path, fps=30)

    while True:
        cup = p.loadURDF("egad_eval_set/model.urdf", [0, -0.1, 0.2], p.getQuaternionFromEuler([math.pi/2, 0., 0]), useFixedBase=True)
        timestep = 0

        while True:
            p.stepSimulation()
            p.setGravity(0,0,-9.8)
            time.sleep(1./240.)
            timestep = timestep + 1



            #pos, orn = p.getBasePositionAndOrientation(cup)
            pos = [0,-0.12,0.415]
            if timestep == 0:
                visualize_data = []
                tactileColor_tmp, _ = gelsight.render()
                visionColor_tmp, _ = cam.get_image()
                visualize_data.append(_align_image(tactileColor_tmp[0], visionColor_tmp))

            ## for seq data
                vision_size, tactile_size = visionColor_tmp.shape, tactileColor_tmp[0].shape
                video_path = os.path.join(save_dir, "demo.mp4")
                rec = utils.video_recorder(vision_size, tactile_size, path=video_path, fps=30)


            elif timestep < 400:

                ori = p.getQuaternionFromEuler([math.pi,0,0])
                joint_poses = panda.calcJointLocation([pos[0], pos[1], pos[2]+0.2], ori )
                panda.setArm(joint_poses)
                panda.setGripper(0.04)

                
            elif timestep>400 and timestep <600:

                ori = p.getQuaternionFromEuler([math.pi,0.,0.])
                joint_poses = panda.calcJointLocation(pos, ori )
                panda.setArm(joint_poses)
                panda.setGripper(0.04)

            elif timestep>600 and timestep <700:
                tactileColor_tmp, depth = gelsight.render()
                visionColor_tmp, _ = cam.get_image()
                visualize_data.append(_align_image(tactileColor_tmp[0], visionColor_tmp))

                ori = p.getQuaternionFromEuler([math.pi,0,0.])
                joint_poses = panda.calcJointLocation(pos, ori )
                panda.setArm(joint_poses)
                panda.setGripper(0.00)
            
            elif timestep>800 and timestep <1000:
                tactileColor_tmp, depth = gelsight.render()
                visionColor_tmp, _ = cam.get_image()
                visualize_data.append(_align_image(tactileColor_tmp[0], visionColor_tmp))
                
                for i in range(1000):
                    angle = math.pi/6 * i/1000
                    ori = p.getQuaternionFromEuler([math.pi,angle,0.])
                    joint_poses = panda.calcJointLocation(pos, ori )
                    panda.setArm(joint_poses)
                    
            elif timestep == 1000:
                        
                tactileColor_tmp, depth = gelsight.render()
                visionColor_tmp, _ = cam.get_image()
                visualize_data.append(_align_image(tactileColor_tmp[0], visionColor_tmp))

            if timestep % 3 == 0 and timestep >0:


                tactileColor_tmp, depth = gelsight.render()
                visionColor, visionDepth = cam.get_image()
                rec.capture(visionColor.copy(), tactileColor_tmp[0].copy())

                cv2.imwrite(save_dir + '/'+str(int(timestep/3)) +'.png', tactileColor_tmp[0])

 

if __name__ == "__main__":
    run()
