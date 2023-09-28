from itertools import combinations
import argparse

import torch
from torch.autograd import Variable
import numpy as np
import scipy.misc
import scipy.io as sio
import heapq

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
    shortest_path = np.empty((0, 9)) #each step has 9 dimensions
    shortest_path_latent = np.empty((0,2)) #each latent step has 2 dimensions
    t = np.linspace(0, 1, steps)
    
    
    # print(idx_src)
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
    start_latent_point = possible_lat[idx_src-possible_grasps[0][2].long()]  
    end_latent_point = possible_lat[idx_dst-possible_grasps[0][2].long()] 
    print("start points", start_latent_point, "\n")
    print("end points",end_latent_point, "\n")
    
    current_latent_point = start_latent_point  # Initialize the current point
    real_start_point = cvae_model.decode(current_latent_point, conditional).detach()


    # Create a set to keep track of visited nodes, uses latent space points
    visited_latent_points = set()
    
    # Create a priority queue with (priority, node) tuples, uses latent space points
    priority_queue = [(0, start_latent_point)]
    
    # Create a dictionary to store distances from the real space starting point
    distances = {tuple(real_start_point.numpy()): 0}
    
    while priority_queue:
        #print("current prio queue", priority_queue,"\n")
        # Dequeue the node with the lowest priority
        current_priority, current_latent_point_tuple = heapq.heappop(priority_queue)
        
        current_latent_point = torch.tensor(current_latent_point_tuple)
    
        # Check if we've reached the ending point
        if torch.equal(current_latent_point, end_latent_point):
            break
    
        # Mark the current node as visited
        visited_latent_points.add(current_latent_point)
    
        # Iterate over neighboring latent points with the same object size/shape
        for neighbor in possible_lat:
            if neighbor not in visited_latent_points:
                # Calculate the tentative distance based on real space distances
                tentative_distance = current_priority + calculate_distance(neighbor, current_latent_point,conditional)
                real_neighbor = tuple(cvae_model.decode(neighbor, conditional).detach().numpy())

                # If this tentative distance is shorter than the current distance, update it
                if real_neighbor not in distances or tentative_distance < distances[real_neighbor]:
                    distances[real_neighbor] = tentative_distance
                    
                    #Preventing the calculations to just merge into direct pathing
                    if torch.equal(neighbor,end_latent_point):
                        priority = current_priority +150
                        #print("end and neighbor are equals, priority set to 100")
                    else:
                    # Calculate the priority (f = a*g + b*h), a>b = prios direct pathing, b>a prios non direct
                        priority = 0.2*tentative_distance + 1.2*heuristic(neighbor, end_latent_point,conditional)
                        #print("end and neighbor are different, adding ", priority, "prio")
                    # Enqueue the neighbor with its priority
                    heapq.heappush(priority_queue, (priority, tuple(neighbor)))
                    
    #print("distnace dict",distances,"\n")
    # Backtrack to find the shortest path
    # print("Backtracking\n")
    current_latent_point = end_latent_point
    
    while not torch.equal(current_latent_point, start_latent_point):
        real_current_point = cvae_model.decode(current_latent_point, conditional).detach()  
        shortest_path_latent = np.vstack([shortest_path_latent, current_latent_point.numpy().astype('float64')])
        shortest_path = np.vstack([shortest_path, real_current_point.numpy().astype('float64')])
    
        # Find the neighbor with the lowest priority (f = (1-hw)*g + (hw)*h)
        min_priority = float('inf')
        next_latent_point = None
    
        for neighbor in possible_lat:
            neighbor_np = neighbor.numpy()
            if not any(np.array_equal(neighbor_np, point.numpy()) for point in visited_latent_points):
                # print("testing: ",neighbor, "\n")
                real_neighbor = tuple(cvae_model.decode(neighbor, conditional).detach().numpy())
                
                #print("Distances:", distances)
                #print("Real Neighbor:", real_neighbor)
                if torch.equal(neighbor,current_latent_point):# or torch.equal(neighbor,start_latent_point):
                    # print("same points 1\n")
                    neighbor_priority = 0.5*distances[real_neighbor] + 1*heuristic(neighbor, start_latent_point,conditional)+150
                else:
                    neighbor_priority = 0.5*distances[real_neighbor] + 1*heuristic(neighbor, start_latent_point,conditional)
                    # print("new point 1\n")
                if neighbor_priority < min_priority:
                    # print("updated min prio 1\n")
                    min_priority = neighbor_priority
                    next_latent_point = neighbor
        
        if next_latent_point is None:
            min_priority = float('inf')
            
            for neighbor in possible_lat:
                real_neighbor = tuple(cvae_model.decode(neighbor, conditional).detach().numpy())
                # print("testing: ",neighbor, "\n")
                
                if torch.equal(neighbor,current_latent_point):# or torch.equal(neighbor,start_latent_point): #this is needed due to float point calculations
                    neighbor_priority = 0.5*distances[real_neighbor] + 1*heuristic(neighbor, start_latent_point,conditional)+150
                    # print("same points 2\n")
                else:
                    neighbor_priority = 0.5*distances[real_neighbor] + 1*heuristic(neighbor, start_latent_point,conditional)
                    # print("new point 2\n")
                    
                if neighbor_priority < min_priority:
                    # print("updated min prio 2\n")
                    min_priority = neighbor_priority
                    next_latent_point = neighbor
                    
                    
        current_latent_point = next_latent_point
        visited_latent_points.add(current_latent_point)
        # print("moving to: ", current_latent_point)
        # print("moving forward")
        # print("\n")
        

    shortest_path = np.vstack([shortest_path, real_start_point.numpy().astype('float64')])
    shortest_path_latent = np.vstack([shortest_path_latent, start_latent_point.numpy().astype('float64')])
    shortest_path = shortest_path[::-1]  # Reverse the path to start from the beginning
    shortest_path_latent = shortest_path_latent[::-1]
    print("shortest path:", shortest_path, "\n")
    print("shortest latent path:", shortest_path_latent, "\n")


# At this point, 'shortest_path' contains the path of latent points from start to end
       
    dist = np.zeros(len(shortest_path[1]))
    for i in range(len(shortest_path[1])):
        for j in range(len(shortest_path)-1):
            dist[i] += np.sqrt((shortest_path[j][i] - shortest_path[j+1][i])**2)
    
    
    dist_total = np.sum(dist)
                        
    print("pathing distances on each dimension")
    print(dist)
    print("\n")
    print("total path distance")
    print(dist_total)
    print("\n")
    # print("average distance per step")
    # print(dist_total/len(trajectory))
    # print("\n")
    
    return shortest_path, dist_total

# Define heuristic function (Euclidean distance)
def heuristic(node, end_node,conditional):
    # Calculate the estimated remaining cost in real space
    return calculate_distance(node, end_node,conditional)

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
    traj, total_dist = get_trajectory(p[0], p[1])
    trajectories.append((traj,total_dist))
    
grasp_idxs = np.asarray(pairs)
model_name = args.model.split('/')[-1]
save_trajectories(grasp_idxs, trajectories, model_name, n_steps,
        '')
