"""
Validation file for MPC control
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


def run():
    database_path = 'egad_eval_set/'      # Data path

    cid = p.connect(p.GUI)  # Connect to server
    env = SimEnv(p, database_path) # Initialize simulation environment
    #INitialize Panda robot
    panda = panda_sim.PandaSimAuto(p, [0, -0.5, 0])     #[0, -0.5, 0]
    pandaID = panda.panda

    GRASP_STATE = False
    grasp_config = {'x':0, 'y':0, 'z':0.01, 'angle':0, 'width':0.16}

    img_path = 'img/img_urdf'

    all_num = 0    
    obj_nums = 1   
    start_idx = 0   

    idx = start_idx

    cnt = 0
    traj_times = 500

    while True:
        x = random.uniform(-0.2,0.2)
        y = random.uniform(-0.2,0.2)
        cup = p.loadURDF("egad_eval_set/model.urdf", [x, y,0])

        idx += obj_nums
        timestep = 0
        action_list = []
        endpose_list = []

        while True:
            p.stepSimulation()
            p.setGravity(0,0,-9.8)
            time.sleep(1./240.)
            timestep = timestep + 1

            pos, orn = p.getBasePositionAndOrientation(cup)

            # Dectect Key press
            keys = p.getKeyboardEvents()
            if ord('1') in keys and keys[ord('1')]&p.KEY_WAS_TRIGGERED: 
                # Rendering image
               env.renderURDFImage(save_path=img_path)
            if ord('2') in keys and keys[ord('2')]&p.KEY_WAS_TRIGGERED:
                GRASP_STATE = True

            if GRASP_STATE:
                # Grasping as predicted
                if panda.test([pos[0], pos[1], pos[2]+0.03], grasp_config['angle'], (grasp_config['width'])/2):
                    #GRASP_STATE = False
                    pass


            pos, orn = p.getBasePositionAndOrientation(cup)
            # print(pos)


 

if __name__ == "__main__":
    run()
