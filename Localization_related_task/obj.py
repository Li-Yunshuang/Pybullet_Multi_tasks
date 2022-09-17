############OBJ Package##############
import os
import datetime
import sys
from turtle import end_fill
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

from objfolder import AudioNet_utils
from objfolder import AudioNet_model


from objfolder.utils import *
#import Matplotlib.pyplot as plt 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def AudioNet_eval(contact_points, force, num, syns=False):

    checkpoint = torch.load("demo/ObjectFile.pth")

    normalizer_dic = checkpoint['AudioNet']['normalizer']
    gains_f1_min = normalizer_dic['f1_min']
    gains_f1_max = normalizer_dic['f1_max']
    gains_f2_min = normalizer_dic['f2_min']
    gains_f2_max = normalizer_dic['f2_max']
    gains_f3_min = normalizer_dic['f3_min']
    gains_f3_max = normalizer_dic['f3_max']
    xyz_min = normalizer_dic['xyz_min']
    xyz_max = normalizer_dic['xyz_max']
    freqs = checkpoint['AudioNet']['frequencies']
    damps = checkpoint['AudioNet']['dampings']


    if syns==False:
        cnt = 0
        

    #forces = np.array([[1., 1., 1.]])
    forces = force
    xyz = np.array([[contact_points[0], contact_points[1], contact_points[2]]])

    # xyz = np.load(args.audio_vertices_file_path).reshape((-1, 3))
    # normalize xyz to [-1, 1]
    xyz = (xyz - xyz_min) / (xyz_max - xyz_min)

    N = xyz.shape[0]
    G = freqs.shape[0]

    embed_fn, input_ch = AudioNet_model.get_embedder(10, 0)
    model = AudioNet_model.AudioNeRF(D=8, input_ch=input_ch, output_ch=G)
    state_dic = checkpoint['AudioNet']["model_state_dict"]
    state_dic = AudioNet_utils.strip_prefix_if_present(state_dic, 'module.')
    model.load_state_dict(state_dic)
    model = nn.DataParallel(model).to(device)
    model.eval()

    preds_gain_x = torch.zeros((N, G)).to(device)
    preds_gain_y = torch.zeros((N, G)).to(device)
    preds_gain_z = torch.zeros((N, G)).to(device)

    batch_size = 1024

    for i in trange(N // batch_size + 1):
        curr_x = torch.Tensor(xyz[i*batch_size:(i+1)*batch_size]).to(device)
        curr_y = torch.Tensor(xyz[i*batch_size:(i+1)*batch_size]).to(device)
        curr_z = torch.Tensor(xyz[i*batch_size:(i+1)*batch_size]).to(device)
        embedded_x = embed_fn(curr_x)
        embedded_y = embed_fn(curr_y)
        embedded_z = embed_fn(curr_z)
        results_x, results_y, results_z = model(embedded_x, embedded_y, embedded_z)

        preds_gain_x[i*batch_size:(i+1)*batch_size] = results_x
        preds_gain_y[i*batch_size:(i+1)*batch_size] = results_y
        preds_gain_z[i*batch_size:(i+1)*batch_size] = results_z

    preds_gain_x = preds_gain_x * (gains_f1_max - gains_f1_min) + gains_f1_min
    preds_gain_y = preds_gain_y * (gains_f2_max - gains_f2_min) + gains_f2_min
    preds_gain_z = preds_gain_z * (gains_f3_max - gains_f3_min) + gains_f3_min
    preds_gain = torch.cat((preds_gain_x[:, None, :], preds_gain_y[:, None, :], preds_gain_z[:, None, :]), 1)

    freqs = torch.Tensor(freqs).to(device)
    damps = torch.Tensor(damps).to(device)

    # testsavedir = args.audio_results_dir
    # os.makedirs(testsavedir, exist_ok=True)
    testsavedir = './results/audio/'
    os.makedirs(testsavedir, exist_ok=True)

    for i in trange(N):
        preds_gain_x_i = preds_gain[i, 0, :]
        preds_gain_y_i = preds_gain[i, 1, :]
        preds_gain_z_i = preds_gain[i, 2, :]
        force_x, force_y, force_z = forces[i]
        combined_preds_gain = force_x * preds_gain_x_i + force_y * preds_gain_y_i + force_z * preds_gain_z_i
        combined_preds_gain = combined_preds_gain.unsqueeze(0)
        modal_fir = torch.unsqueeze(ddsp.get_modal_fir(combined_preds_gain, freqs, damps), axis=1)
        impulse = torch.reshape(torch.Tensor(scipy.signal.unit_impulse(44100*3)).to(device), (1, -1)).repeat(modal_fir.shape[0], 1)
        result = ddsp.fft_convolve(impulse, modal_fir)
        signal = result[0, :].detach().cpu().numpy()
        signal = signal / np.abs(signal).max()
        # write wav file
        output_path = os.path.join(testsavedir, str(num) + '.wav')
        write(output_path, 44100, signal.astype(np.float32))


def TouchNet_eval(contact_points, num, path):

    checkpoint = torch.load("demo/ObjectFile.pth")

    rotation_max = 15
    depth_max = 0.04
    depth_min = 0.0339
    displacement_min = 0.0005
    displacement_max = 0.0020
    depth_max = 0.04
    depth_min = 0.0339
    rgb_width = 120
    rgb_height = 160
    network_depth = 8

    #TODO load object...
    vertex_min = checkpoint['TouchNet']['xyz_min']     # obj xyz 
    vertex_max = checkpoint['TouchNet']['xyz_max']     

    vertex_coordinates = np.array([[contact_points[0], contact_points[1], contact_points[2]]])
    N = vertex_coordinates.shape[0]     # Single contact point
    #gelinfo_data = np.load(args.touch_gelinfo_file_path)
    gelinfo_data = np.array([[0., 0. , 0.001]])
    theta, phi, displacement = gelinfo_data[:, 0], gelinfo_data[:, 1], gelinfo_data[:, 2]
    phi_x = np.cos(phi)
    phi_y = np.sin(phi)

    # normalize theta to [-1, 1]
    theta = (theta - np.radians(0)) / (np.radians(rotation_max) - np.radians(0))

    #normalize displacement to [-1,1]
    displacement_norm = (displacement - displacement_min) / (displacement_max - displacement_min)

    #normalize coordinates to [-1,1]
    vertex_coordinates = (vertex_coordinates - vertex_min) / (vertex_max - vertex_min)

    #initialize horizontal and vertical features
    w_feats = np.repeat(np.repeat(np.arange(rgb_width).reshape((rgb_width, 1)), rgb_height, axis=1).reshape((1, 1, rgb_width, rgb_height)), N, axis=0)
    h_feats = np.repeat(np.repeat(np.arange(rgb_height).reshape((1, rgb_height)), rgb_width, axis=0).reshape((1, 1, rgb_width, rgb_height)), N, axis=0)
    #normalize horizontal and vertical features to [-1, 1]
    w_feats_min = w_feats.min()
    w_feats_max = w_feats.max()
    h_feats_min = h_feats.min()
    h_feats_max = h_feats.max()
    w_feats = (w_feats - w_feats_min) / (w_feats_max - w_feats_min)
    h_feats = (h_feats - h_feats_min) / (h_feats_max - h_feats_min)
    w_feats = torch.FloatTensor(w_feats)
    h_feats = torch.FloatTensor(h_feats)

    theta = np.repeat(theta.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    phi_x = np.repeat(phi_x.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    phi_y = np.repeat(phi_y.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    displacement_norm = np.repeat(displacement_norm.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    vertex_coordinates = np.repeat(vertex_coordinates.reshape((N, 1, 3)), rgb_width * rgb_height, axis=1)

    data_wh = np.concatenate((w_feats, h_feats), axis=1)
    data_wh = np.transpose(data_wh.reshape((N, 2, -1)), axes=[0, 2, 1])
    #Now get final feats matrix as [x, y, z, theta, phi_x, phi_y, displacement, w, h]
    data = np.concatenate((vertex_coordinates, theta, phi_x, phi_y, displacement_norm, data_wh), axis=2).reshape((-1, 9))

    #checkpoint = torch.load(args.object_file_path)
    embed_fn, input_ch = TouchNet_model.get_embedder(10, 0)
    model = TouchNet_model.NeRF(D = network_depth, input_ch = input_ch, output_ch = 1)
    state_dic = checkpoint['TouchNet']['model_state_dict']
    state_dic = TouchNet_utils.strip_prefix_if_present(state_dic, 'module.')
    model.load_state_dict(state_dic)
    model = nn.DataParallel(model).to(device)
    model.eval()

    preds = np.empty((data.shape[0], 1))

    batch_size = 1024

    testsavedir = path
    os.makedirs(testsavedir, exist_ok=True)

    for i in trange(data.shape[0] // batch_size + 1):
        inputs = torch.Tensor(data[i*batch_size:(i+1)*batch_size]).to(device)
        embedded = embed_fn(inputs)
        results = model(embedded)
        preds[i*batch_size:(i+1)*batch_size, :] = results.detach().cpu().numpy()

    preds = preds  * (depth_max - depth_min) + depth_min
    preds = np.transpose(preds.reshape((N, -1, 1)), axes = [0, 2, 1]).reshape((N, rgb_width, rgb_height))
    taxim = TaximRender("./calibs/")
    for i in trange(N):
        height_map, contact_map, tactile_map = taxim.render(preds[i], displacement[i])
        tactile_map = Image.fromarray(tactile_map.astype(np.uint8), 'RGB')
        filename = os.path.join(testsavedir, '{}.png'.format(num))
        tactile_map.save(filename)
        tactile_map.show()
        #plt.imshow(tactile_map)