# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 21:47:53 2023

@author: jorge
"""

from itertools import combinations
import argparse

import torch
from torch.autograd import Variable
import numpy as np
import scipy.misc
import scipy.io as sio

from cvae import CVAE
from utils import one_hot, get_lininter, get_CVAE, save_trajectories
from data_loader import load_data, annotate_grasps

import random 

def parse_args():
    parser = argparse.ArgumentParser(description='Train AE for Shadow Hand')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--kl', type=float, default=1.0, metavar='N',
                        help='kl weight')
    parser.add_argument('--data', type=str, default='data/', metavar='N',
                        help='directory with grasp data')
    args = parser.parse_args()
    return args


args = parse_args()

grasps, labels = load_data(args.data, robot='icub')

grasps, grasp_type, object_type, object_size = annotate_grasps(grasps, labels)

pairs = np.load('data/final_pairs.npy')
print (f'Checking {len(pairs)} pairs!')


for p in range(len(pairs)):
    start_point = pairs[p][0]
    end_point = pairs[p][1]
    
    start_point_object = labels[start_point][0]
    end_point_object = labels[end_point][0]
    
    if start_point_object != end_point_object:
        print(f'error in pair {p}')
        continue

        
    
    
    
    

    