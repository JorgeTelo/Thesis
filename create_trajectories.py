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
random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Sample grasps from a VAE')
    parser.add_argument('--model', type=str, metavar='N',
                        help='model path')
    parser.add_argument('--data', type=str, default='data/', metavar='N',
                        help='original dataset')
    parser.add_argument('--steps', type=int, default=35, metavar='N',
                        help='number of steps in interpolation')
    args = parser.parse_args()
    return args

def get_conditional(obj_shape, obj_size):
    """To numpy array"""
    obj_shape = np.c_[obj_shape]
    obj_size = np.c_[obj_size]

    """To torch tensor"""
    obj_shape = torch.from_numpy(obj_shape).to(dtype=torch.float)
    obj_size = torch.from_numpy(obj_size).to(dtype=torch.float)

    """One hot encoding"""
    obj_shape = one_hot(obj_shape.long(), class_size=3)

    conditional = torch.cat((obj_shape, obj_size), dim=-1)
    return conditional

def get_trajectory(idx_src, idx_dst, steps=35, obj_type=3, obj_size=1.0):
    trajectory = np.empty((0, 9))
    t = np.linspace(0, 1, steps)
    #print("t: ", t)

    print (labels[idx_src])
    # print (object_type[idx_src])
    # print (object_size[idx_src])
    # print (object_type[idx_dst])
    # print (object_size[idx_dst] - 1.0)
    cond_size = object_size[idx_dst] - 1.0
    # if object_size[idx_dst] == 0.0:
    #     cond_size -= 0.5
    # elif object_size[idx_dst] == -1.0:
    #     cond_size -= 1.0

    print (cond_size)
    print ('\n')
    conditional = torch.cat((object_type[idx_src], cond_size), dim=-1)

    for j in range(steps):
        lat = get_lininter(X_lat[idx_src], X_lat[idx_dst], t[j])
        inter_grasp = cvae_model.decode(lat, conditional).detach()            
        trajectory = np.vstack([trajectory, inter_grasp.numpy().astype('float64')])

    return trajectory

args = parse_args()
n_steps = args.steps

# Load data
grasps, labels = load_data(args.data, robot='icub')

grasps, grasp_type, object_type, object_size = annotate_grasps(grasps, labels)

# indices_map = list(np.arange(grasps.shape[0])[idxs_n])

# To torch tensors
grasps = torch.from_numpy(grasps).float()
grasps_tosee = grasps.numpy()
object_size = torch.from_numpy(object_size).float().unsqueeze(-1)
object_type = torch.from_numpy(object_type).float()
# One hot encoding
object_type = one_hot(object_type.long(), class_size=3)

y = torch.cat((object_type, object_size), dim=-1)

cvae_model = get_CVAE(args.model) 
recon_grasps, X_lat, _ = cvae_model(grasps, y)

trajectories = np.empty((0, n_steps, 9))


pairs = np.load('data/final_pairs.npy')
traj_cost = np.empty(len(pairs))
print (f'Generating {len(pairs)} pairs!')
# pairs = np.load('pairs_test.npy')
pair_index = 0

for p in pairs:
    #print("p:", p)
    traj = get_trajectory(p[0], p[1], n_steps)
    #print("traj: ", traj)
    trajectories = np.vstack([trajectories, np.expand_dims(traj, axis=0)])
    traj_cost[pair_index] = np.sum(traj)
    pair_index += 1

grasp_idxs = np.asarray(pairs)
model_name = args.model.split('/')[-1]
save_trajectories(grasp_idxs, trajectories, model_name, n_steps,
        '')
