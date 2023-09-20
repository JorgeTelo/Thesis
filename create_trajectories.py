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
    parser.add_argument('--steps', type=int, default=10, metavar='N',
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
    trajectory = np.empty((0, 9)) #each step has 9 dimensions
    t = np.linspace(0, 1, steps)
    
    
    print(idx_src)
    # print (labels[idx_src])
    # print (object_type[idx_src])
    # print (object_size[idx_src])
    # print (object_type[idx_dst])
    # print (object_size[idx_dst] - 1.0)
    cond_size = object_size[idx_dst] - 1.0
    # if object_size[idx_dst] == 0.0:
    #     cond_size -= 0.5
    # elif object_size[idx_dst] == -1.0:
    #     cond_size -= 1.0
    # print('latent source',X_lat[idx_src])
    # print('\n')
    # print('latent end',X_lat[idx_dst])
    # print('\n')
    # print (cond_size)
    # print ('\n')
    conditional = torch.cat((object_type[idx_src], cond_size), dim=-1)
    possible_grasps, possible_lat = remove_unwanted_objects(idx_src)
    print('    possible grasps ', len(possible_grasps), '\n')
    #print(possible_grasps)


    # Define your starting and ending points in the latent space
    start_latent_point = possible_lat[idx_src-possible_grasps[0][2].long()]  # Replace with your actual starting point
    end_latent_point = possible_lat[idx_dst-possible_grasps[0][2].long()] # Replace with your actual ending point
    print("start points", start_latent_point, "\n")
    print("end points",end_latent_point, "\n")
    
    visited_latent_points = set()  # To keep track of visited latent points
    current_latent_point = start_latent_point  # Initialize the current point
    
    path = cvae_model.decode(current_latent_point, conditional).detach()      # Initialize the path with the starting point
    trajectory = np.vstack([trajectory, path.numpy().astype('float64')])
    
    
    while not torch.equal(current_latent_point, end_latent_point):
        # Find the nearest unvisited neighbor
        nearest_neighbor = None
        min_distance = 1e10
    
        for candidate_point in possible_lat:
            
            # Need to use numpy arrays to correctly use loops because tensors are not suitable for sets.....
            #print("candidate: ",candidate_point, "\n")
            candidate_point_np = candidate_point.numpy()
            
            
            if not any(np.array_equal(candidate_point_np, point.numpy()) for point in visited_latent_points):
                distance = calculate_distance(candidate_point, current_latent_point,conditional)
                if distance < min_distance:
                    min_distance = distance
                    nearest_neighbor = candidate_point
                    #print("new nearest neighbour: ", nearest_neighbor, "\n")
        
        # Update the current point and add it to the path
        current_latent_point = nearest_neighbor
        real_current_toadd = cvae_model.decode(current_latent_point, conditional).detach() 
        trajectory = np.vstack([trajectory, real_current_toadd.numpy().astype('float64')])
    
        # Mark the current point as visited
        visited_latent_points.add(current_latent_point)

# At this point, 'path' contains the path of latent points from start to end
    # for j in range(steps):

    #     lat = get_lininter(X_lat[idx_src], X_lat[idx_dst], t[j])
    #     inter_grasp = cvae_model.decode(lat, conditional).detach()            
    #     trajectory = np.vstack([trajectory, inter_grasp.numpy().astype('float64')])
        
        
    dist = np.zeros(len(trajectory[1]))
    for i in range(len(trajectory[1])):
        for j in range(len(trajectory)-1):
            dist[i] += np.sqrt((trajectory[j][i] - trajectory[j+1][i])**2)
    
    
    dist_total = np.sum(dist)
                        
    print("pathing distances on each dimension")
    print(dist)
    print("\n")
    print("total path distance")
    print(dist_total)
    print("\n")
    
    return trajectory,dist_total

def calculate_distance(candidate_point, current_latent_point,conditional):
    
    real_candidate = cvae_model.decode(candidate_point, conditional).detach()
    real_current = cvae_model.decode(current_latent_point, conditional).detach()
    
    # Calculate the squared differences in each dimension
    squared_diff = (real_candidate.sum() - real_current.sum()) ** 2
    
    # Sum the squared differences and take the square root
    distance = np.sqrt((squared_diff))
    
    return distance

def remove_unwanted_objects(idx_src):
    desired_object = labels[idx_src][0]
    print('desired object', desired_object)
    
    flatten_label = np.delete(labels,1,axis=1)
    
    unwanted_indexes = np.where(flatten_label != desired_object)
    wanted_indexes = np.where(flatten_label == desired_object)
    wanted_indexes = wanted_indexes[0].T.reshape(len(wanted_indexes[0]),1)
    
    grasps_touse = np.delete(labels,unwanted_indexes[0],axis=0)
    grasps_touse = np.append(grasps_touse, wanted_indexes,axis=1)
    lat_touse = np.delete(X_lat.detach().numpy(),unwanted_indexes[0],axis=0)
    grasps_touse_tensor = torch.from_numpy(grasps_touse).float()
    lat_touse_tensor = torch.from_numpy(lat_touse).float()
    

    
    return grasps_touse_tensor, lat_touse_tensor

args = parse_args()
n_steps = args.steps

# Load data
grasps, labels = load_data(args.data, robot='icub')

grasps, grasp_type, object_type, object_size = annotate_grasps(grasps, labels)

# indices_map = list(np.arange(grasps.shape[0])[idxs_n])

# To torch tensors
grasps = torch.from_numpy(grasps).float()
object_size = torch.from_numpy(object_size).float().unsqueeze(-1)
object_type = torch.from_numpy(object_type).float()
# One hot encoding
object_type = one_hot(object_type.long(), class_size=3)

y = torch.cat((object_type, object_size), dim=-1)

cvae_model = get_CVAE(args.model) 
recon_grasps, X_lat, _ = cvae_model(grasps, y)

X_lat_toread = X_lat.detach().numpy()

trajectories = list()

pairs = np.load('data/final_pairs.npy')
print (f'Generating {len(pairs)} pairs!')
# pairs = np.load('pairs_test.npy')

for p in pairs:
    traj,total_dist = get_trajectory(p[0], p[1])
    trajectories.append((traj,total_dist))
    
    

grasp_idxs = np.asarray(pairs)
model_name = args.model.split('/')[-1]
save_trajectories(grasp_idxs, trajectories, model_name, n_steps,
        '')
