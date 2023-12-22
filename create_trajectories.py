from itertools import combinations
import argparse
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import numpy as np
import scipy.misc
import scipy.io as sio
import heapq

from cvae import CVAE
from utils import one_hot, get_lininter, get_CVAE, save_trajectories
from data_loader import load_data, annotate_grasps
from scipy.spatial.distance import pdist, squareform, cdist

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

def get_trajectory(idx_src, idx_dst, steps=9):
    trajectory = np.empty((0, len(grasps[1]))) #each step has 9 dimensions #each step has 9 dimensions
    
    trajectory_latent = np.empty((0,lat_dim)) #each latent step has 2 dimensions
    t = np.linspace(0, 1, steps)
    test_trajectory = np.empty((0, len(grasps[1]))) #each step has 9 dimensions #each step has 9 dimensions
    test_trajectory_latent = np.empty((0,lat_dim)) #each latent step has 2 dimensions
    straight_trajectory = np.empty((0, len(grasps[1]))) #each step has 9 dimensions #each step has 9 dimensions
    straight_trajectory_latent = np.empty((0,lat_dim)) #each latent step has 2 dimensions
    
    shortest_path = np.empty((0, len(grasps[1]))) #each step has 9 dimensions
    shortest_path_latent = np.empty((0,lat_dim)) #each latent step has 2 dimensions
    
    astar_latent_path = np.empty((0, len(grasps[1]))) #each step has 9 dimensions
    astar_latent_path_latent = np.empty((0,lat_dim)) #each latent step has 2 dimensions
    
    # print(idx_src)
    # print (labels[idx_src])
    # print (object_type[idx_src])
    # print (object_size[idx_src])
    # print (object_type[idx_dst])
    # print (object_size[idx_dst] - 1.0)
    obj_type = labels[idx_src][0]
    obj_type_start = labels[idx_src][0]
    obj_type_end = labels[idx_dst][0]
    
    if obj_type_start != obj_type_end:
        return 0,0,0,0,0,0,0,0,0
    obj_used.append(obj_type)
    
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
    X_lat_dummy = X_lat.detach()
    print('    possible grasps ', len(possible_grasps), '\n')
    #print(possible_grasps)


    # Define your starting and ending points in the latent space
    start_latent_point =  X_lat[idx_src].detach()
    end_latent_point = X_lat[idx_dst].detach()
    print("start points", start_latent_point, "\n")
    print("end points",end_latent_point, "\n")
    
    
    # ##### This part is looking through real space neighbors ####
    current_latent_point = start_latent_point  # Initialize the current point
    real_start_point = cvae_model.decode(current_latent_point, conditional, mean_tensor, std_tensor, obj_type).detach()
    
    visited_latent_points = set()  # To keep track of visited latent points
    current_latent_point = start_latent_point  # Initialize the current point
    visited_latent_points.add(current_latent_point)
    
    path = cvae_model.decode(current_latent_point, conditional, mean_tensor, std_tensor, obj_type).detach()   
    trajectory_latent = np.vstack([trajectory_latent, current_latent_point])# Initialize the path with the starting point
    trajectory = np.vstack([trajectory, path.numpy().astype('float64')])
    
    print("Starting first loop\n")
    while not torch.equal(current_latent_point, end_latent_point):
        # Find the nearest unvisited neighbor
        nearest_neighbor = None
        min_distance = 1e10
    
        for candidate_point in possible_lat:
            
            # Need to use numpy arrays to correctly use loops because tensors are not suitable for sets.....
            #print("candidate: ",candidate_point, "\n")
            candidate_point_np = candidate_point.numpy()
            
            
            if not any(np.array_equal(candidate_point_np, point.numpy()) for point in visited_latent_points):
                distance = calculate_real_distance(candidate_point, current_latent_point,conditional, mean_tensor, std_tensor, obj_type)
                if distance < min_distance:
                    min_distance = distance
                    nearest_neighbor = candidate_point
                    # print("new nearest neighbour: ", nearest_neighbor, "\n")
        
        # Update the current point and add it to the path
        current_latent_point = nearest_neighbor
        trajectory_latent = np.vstack([trajectory_latent, current_latent_point])
        real_current_toadd = cvae_model.decode(current_latent_point, conditional, mean_tensor, std_tensor, obj_type).detach()
        trajectory = np.vstack([trajectory, real_current_toadd.numpy().astype('float64')])
    
        # Mark the current point as visited
        visited_latent_points.add(current_latent_point)
        

    
    # print("shortest path:", trajectory, "\n")
    # print("shortest latent path:", trajectory_latent, "\n")


# # At this point, 'shortest_path' contains the path of latent points from start to end
       
    dist = np.zeros(len(trajectory[1]))
    for i in range(len(trajectory[1])):
        for j in range(len(trajectory)-1):
            dist[i] += np.sqrt((trajectory[j][i] - trajectory[j+1][i])**2)
    
    
    dist_total = np.sum(dist)
           
    print("Pathing with real neighbors")             
    # print("pathing distances on each dimension")
    # print(dist)
    print("\n")
    print("total path distance")
    print(dist_total)
    print("\n")
    print("average distance per step")
    print(dist_total/len(trajectory))
    print("\n")
    
#     # # Plot the latent space in a scatter plot
#     # plt.figure(figsize=(8, 6))
#     # plt.scatter(possible_lat[:, 0], possible_lat[:, 1], c='b', marker='o', s=10)
#     # plt.title('2D Latent Space')
#     # plt.xlabel('Latent Dimension 1')
#     # plt.ylabel('Latent Dimension 2')
#     # plt.plot(trajectory_latent[:, 0], trajectory_latent[:, 1], c='r', marker='o', linestyle='-')
#     # plt.grid(True)
#     # plt.show()
    
#     ###########################################################
#     ###########################################################
#     #### This part is looking neighbors in latent space   #####
    print("Starting 2nd loop\n")
    current_latent_point = start_latent_point  # Initialize the current point
    real_start_point = cvae_model.decode(current_latent_point, conditional, mean_tensor, std_tensor, obj_type).detach()
    
    visited_latent_points = set()  # To keep track of visited latent points
    current_latent_point = start_latent_point  # Initialize the current point
    visited_latent_points.add(current_latent_point)
    
    path = cvae_model.decode(current_latent_point, conditional, mean_tensor, std_tensor, obj_type).detach()   
    test_trajectory_latent = np.vstack([test_trajectory_latent, current_latent_point])# Initialize the path with the starting point
    test_trajectory = np.vstack([test_trajectory, path.numpy().astype('float64')])
    
    
    while not torch.equal(current_latent_point, end_latent_point):
        # Find the nearest unvisited neighbor
        nearest_neighbor = None
        min_distance = 1e10
    
        for candidate_point in possible_lat:
            candidate_point_np = candidate_point.detach().numpy()
            if not any(np.array_equal(candidate_point_np, point.numpy()) for point in visited_latent_points):
                # print("new candidate_point: ", candidate_point, "\n")
                distance = calculate_latent_distance(candidate_point, current_latent_point,conditional)
                if distance < min_distance:
                    min_distance = distance
                    # print("new min dist: ", min_distance, "\n")
                    nearest_neighbor = candidate_point
                    # print("new nearest neighbour: ", nearest_neighbor, "\n")
                    # if torch.equal(nearest_neighbor, end_latent_point):
                        # print("end reached\n")
        
        # Update the current point and add it to the path
        current_latent_point = nearest_neighbor
        # print("moving to:", current_latent_point, "\n")
        test_trajectory_latent = np.vstack([test_trajectory_latent, current_latent_point])
        real_current_toadd = cvae_model.decode(current_latent_point, conditional, mean_tensor, std_tensor, obj_type).detach() 
        test_trajectory = np.vstack([test_trajectory, real_current_toadd.numpy().astype('float64')])
    
        # Mark the current point as visited
        visited_latent_points.add(current_latent_point)
        

    
    # print("shortest path using latent space:", test_trajectory, "\n")
    # print("shortest latent path latent space:", test_trajectory_latent, "\n")


# At this point, 'shortest_path' contains the path of latent points from start to end
       
    test_dist = np.zeros(len(test_trajectory[1]))
    for i in range(len(test_trajectory[1])):
        for j in range(len(test_trajectory)-1):
            test_dist[i] += np.sqrt((test_trajectory[j][i] - test_trajectory[j+1][i])**2)
    
    
    test_dist_total = np.sum(test_dist)
           
    print("Pathing with latent neighbors")             
    # print("pathing distances on each dimension")
    # print(test_dist)
    print("\n")
    print("total path distance")
    print(test_dist_total)
    print("\n")
    # print("average distance per step")
    print(dist_total/len(trajectory))
    print("\n")

    # # Plot the latent space in a scatter plot
    # plt.figure(figsize=(8, 6))
    # # plt.scatter(possible_lat[:, 0], possible_lat[:, 1], c='b', marker='o', s=10)
    # plt.scatter(possible_lat[:, 0], possible_lat[:, 1], c='b', marker='o', s=10)
    # plt.title('2D Latent Space')
    # plt.xlabel('Latent Dimension 1')
    # plt.ylabel('Latent Dimension 2')
    # plt.plot(test_trajectory_latent[:, 0], test_trajectory_latent[:, 1], c='g', marker='o', linestyle='-')
    # plt.plot(trajectory_latent[:, 0], trajectory_latent[:, 1], c='r', marker='o', linestyle='-')
    # plt.grid(True)
    # plt.show()
    
    ## Straight line approach
    print("Starting straight line\n")
    for j in range(steps):
        lat = get_lininter(X_lat[idx_src], X_lat[idx_dst], t[j])
        inter_grasp = cvae_model.decode(lat, conditional, mean_tensor, std_tensor, obj_type).detach()            
        straight_trajectory = np.vstack([straight_trajectory, inter_grasp.numpy().astype('float64')])
    
    straight_dist = np.zeros(len(straight_trajectory[1]))
    for i in range(len(straight_trajectory[1])):
        for j in range(len(straight_trajectory)-1):
            straight_dist[i] += np.sqrt((straight_trajectory[j][i] - straight_trajectory[j+1][i])**2)
    
    
    straight_dist_total = np.sum(straight_dist)
    print("Pathing with straight line")             
    # print("pathing distances on each dimension")
    # print(test_dist)
    print("\n")
    print("total path distance")
    print(straight_dist_total)
    print("\n")
    # print("average distance per step")
    print(straight_dist_total/len(trajectory))
    print("\n")
    
    ##A star approach using real space distances
    print("Starting astar real loop\n")
    
    current_latent_point = start_latent_point  # Initialize the current point
    real_start_point = cvae_model.decode(current_latent_point, conditional, mean_tensor, std_tensor, obj_type).detach()


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
            #print("\nBreaking\n")
            break
    
        # Mark the current node as visited
        visited_latent_points.add(current_latent_point)
    
        # Iterate over neighboring latent points with the same object size/shape
        for neighbor in X_lat_dummy:
            if neighbor not in visited_latent_points:
                # Calculate the tentative distance based on real space distances
                tentative_distance = current_priority + calculate_real_distance(neighbor, current_latent_point,conditional, mean_tensor, std_tensor, obj_type)
                real_neighbor = tuple(cvae_model.decode(neighbor, conditional, mean_tensor, std_tensor, obj_type).detach().numpy())

                # If this tentative distance is shorter than the current distance, update it
                if real_neighbor not in distances or tentative_distance < distances[real_neighbor]:
                    distances[real_neighbor] = tentative_distance
                    
                    # #Preventing the calculations to just merge into direct pathing
                    if torch.equal(neighbor,end_latent_point):
                        priority = 1*tentative_distance
                        # print("end and neighbor are equals, priority set to 100")
                    else:
                    # Calculate the priority (f = a*g + b*h), a>b = prios direct pathing, b>a prios non direct
                        priority = 1*tentative_distance + 1.0*real_heuristic(neighbor, end_latent_point,conditional, mean_tensor, std_tensor, obj_type)
                        #print("end and neighbor are different, adding ", priority, "prio")
                    # Enqueue the neighbor with its priority
                    heapq.heappush(priority_queue, (priority, tuple(neighbor)))
                    
    #print("distnace dict",distances,"\n")
    # Backtrack to find the shortest path
    # print("Backtracking\n")
    current_latent_point = end_latent_point
    
    while not torch.equal(current_latent_point, start_latent_point):
        real_current_point = cvae_model.decode(current_latent_point, conditional, mean_tensor, std_tensor, obj_type).detach()  
        shortest_path_latent = np.vstack([shortest_path_latent, current_latent_point.numpy().astype('float64')])
        shortest_path = np.vstack([shortest_path, real_current_point.numpy().astype('float64')])
    
        # Find the neighbor with the lowest priority (f = (1-hw)*g + (hw)*h)
        min_priority = float('inf')
        next_latent_point = None
    
        for neighbor in X_lat_dummy:
            neighbor_np = neighbor.numpy()
            # if not any(np.array_equal(neighbor_np, point.numpy()) for point in visited_latent_points):
            # print("testing: ",neighbor, "\n")
            real_neighbor = tuple(cvae_model.decode(neighbor, conditional, mean_tensor, std_tensor, obj_type).detach().numpy())
            
            #print("Distances:", distances)
            #print("Real Neighbor:", real_neighbor)
            if torch.equal(neighbor,current_latent_point):# or torch.equal(neighbor,start_latent_point): #this is needed due to float point calculations
                neighbor_priority = 1*distances[real_neighbor] + 1*real_heuristic(neighbor, end_latent_point,conditional,mean_tensor, std_tensor, obj_type)+50
                    # print("same points 2\n")
            else:

                neighbor_priority = 1*distances[real_neighbor] + 1*real_heuristic(neighbor, end_latent_point,conditional, mean_tensor, std_tensor, obj_type)
                # print("new point 1\n")
            if neighbor_priority < min_priority:
                # print("updated min prio 1\n")
                min_priority = neighbor_priority
                next_latent_point = neighbor
                # print("\na\n")
        
        # if next_latent_point is None:
        #     min_priority = float('inf')
            
        #     for neighbor in possible_lat:
        #         real_neighbor = tuple(cvae_model.decode(neighbor, conditional).detach().numpy())
        #         # print("testing: ",neighbor, "\n")
                
        #         if torch.equal(neighbor,current_latent_point):# or torch.equal(neighbor,start_latent_point): #this is needed due to float point calculations
        #             neighbor_priority = 1*real_heuristic(neighbor, start_latent_point,conditional)
        #             # print("same points 2\n")
        #         else:
        #             neighbor_priority = 0.5*distances[real_neighbor] + 1*real_heuristic(neighbor, start_latent_point,conditional)
        #             # print("new point 2\n")
                    
        #         if neighbor_priority < min_priority:
        #             # print("updated min prio 2\n")
        #             min_priority = neighbor_priority
        #             next_latent_point = neighbor
        #             # print("\nb\n")
                    
                    
        current_latent_point = next_latent_point
        visited_latent_points.add(current_latent_point)
        # print("moving to: ", current_latent_point)
        # print("moving forward")
        # print("\n")
        

    shortest_path = np.vstack([shortest_path, real_start_point.numpy().astype('float64')])
    shortest_path_latent = np.vstack([shortest_path_latent, start_latent_point.numpy().astype('float64')])
    shortest_path = shortest_path[::-1]  # Reverse the path to start from the beginning
    shortest_path_latent = shortest_path_latent[::-1]
    # print("shortest path:", shortest_path, "\n")
    # print("shortest latent path:", shortest_path_latent, "\n")


# At this point, 'shortest_path' contains the path of latent points from start to end
       
    astar_dist = np.zeros(len(shortest_path[1]))
    for i in range(len(shortest_path[1])):
        for j in range(len(shortest_path)-1):
            astar_dist[i] += np.sqrt((shortest_path[j][i] - shortest_path[j+1][i])**2)
            

    
    
    
    astar_dist_total = np.sum(astar_dist)
    print("A star pathing\n")
    # print("pathing distances on each dimension")
    # print(astar_dist)
    # print("\n")
    print("total path distance")
    print(astar_dist_total)
    print("\n")
    # print("average distance per step")
    # print(dist_total/len(trajectory))
    # print("\n")
    
    
    ##A star approach using real space distances
    print("Starting astar latent loop\n")
    
    current_latent_point = start_latent_point  # Initialize the current point
    real_start_point = cvae_model.decode(current_latent_point, conditional, mean_tensor, std_tensor, obj_type).detach()


    # Create a set to keep track of visited nodes, uses latent space points
    visited_latent_points = set()
    
    # Create a priority queue with (priority, node) tuples, uses latent space points
    priority_queue = [(0, start_latent_point)]
    
    # Create a dictionary to store distances from the real space starting point
    distances = {tuple(start_latent_point.numpy()): 0}
    
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
        for neighbor in X_lat_dummy:
            if neighbor not in visited_latent_points:
                # Calculate the tentative distance based on real space distances
                tentative_distance = current_priority + calculate_latent_distance(neighbor, current_latent_point,conditional)
                real_neighbor = tuple(cvae_model.decode(neighbor, conditional, mean_tensor, std_tensor, obj_type).detach().numpy())

                # If this tentative distance is shorter than the current distance, update it
                if tuple(neighbor.numpy()) not in distances or tentative_distance < distances[tuple(neighbor.numpy())]:
                    distances[tuple(neighbor.numpy())] = tentative_distance
                    
                    #Preventing the calculations to just merge into direct pathing
                    if torch.equal(neighbor,end_latent_point):
                        priority = 1*tentative_distance
                        #print("end and neighbor are equals, priority set to 100")
                    else:
                    # Calculate the priority (f = a*g + b*h), a>b = prios direct pathing, b>a prios non direct
                        priority = 1*tentative_distance + 1*lat_heuristic(neighbor, end_latent_point,conditional)
                        #print("end and neighbor are different, adding ", priority, "prio")
                    # Enqueue the neighbor with its priority
                    heapq.heappush(priority_queue, (priority, tuple(neighbor)))
                    
    #print("distnace dict",distances,"\n")
    # Backtrack to find the shortest path
    # print("Backtracking\n")
    current_latent_point = end_latent_point
    

    
    while not torch.equal(current_latent_point, start_latent_point):
        real_current_point = cvae_model.decode(current_latent_point, conditional, mean_tensor, std_tensor, obj_type).detach()  
        astar_latent_path_latent = np.vstack([astar_latent_path_latent, current_latent_point.numpy().astype('float64')])
        astar_latent_path = np.vstack([astar_latent_path, real_current_point.numpy().astype('float64')])
    
        # Find the neighbor with the lowest priority (f = (1-hw)*g + (hw)*h)
        min_priority = float('inf')
        next_latent_point = None
    
        for neighbor in X_lat_dummy:
            neighbor_np = neighbor.numpy()
            # if not any(np.array_equal(neighbor_np, point.numpy()) for point in visited_latent_points):
            # print("testing: ",neighbor, "\n")
            real_neighbor = tuple(cvae_model.decode(neighbor, conditional, mean_tensor, std_tensor, obj_type).detach().numpy())
            
            #print("Distances:", distances)
            #print("Real Neighbor:", real_neighbor)
            if torch.equal(neighbor,current_latent_point):# or torch.equal(neighbor,start_latent_point): #this is needed due to float point calculations
                    neighbor_priority = 1*distances[tuple(neighbor.numpy())] + 1*lat_heuristic(neighbor, end_latent_point,conditional)+50
                    # print("same points 2\n")
            else:
                neighbor_priority = 1*distances[tuple(neighbor.numpy())] + 1*lat_heuristic(neighbor, end_latent_point,conditional)
                # print("new point 1\n")
            if neighbor_priority < min_priority:
                # print("updated min prio 1\n")
                min_priority = neighbor_priority
                next_latent_point = neighbor
                # print("i'm b\n")
                # print("\na\n")
        
        # if next_latent_point is None:
        #     min_priority = float('inf')
            
        #     for neighbor in possible_lat:
        #         real_neighbor = tuple(cvae_model.decode(neighbor, conditional).detach().numpy())
        #         # print("testing: ",neighbor, "\n")
                
        #         if torch.equal(neighbor,current_latent_point):# or torch.equal(neighbor,start_latent_point): #this is needed due to float point calculations
        #             neighbor_priority = 0.5*distances[tuple(neighbor.numpy())] + 150 + 1*lat_heuristic(neighbor, start_latent_point,conditional)
        #             # print("same points 2\n")
        #         else:
        #             neighbor_priority = 0.5*distances[tuple(neighbor.numpy())] + 1*lat_heuristic(neighbor, start_latent_point,conditional)
        #             # print("new point 2\n")
                    
        #         if neighbor_priority < min_priority:
        #             # print("updated min prio 2\n")
        #             min_priority = neighbor_priority
        #             next_latent_point = neighbor
                    
        # print("i'm c\n")
        current_latent_point = next_latent_point
        visited_latent_points.add(current_latent_point)
        # print("moving to: ", current_latent_point)
        # print("moving forward")
        # print("\n")
        

    astar_latent_path = np.vstack([astar_latent_path, real_start_point.numpy().astype('float64')])
    astar_latent_path_latent = np.vstack([astar_latent_path_latent, start_latent_point.numpy().astype('float64')])
    astar_latent_path = astar_latent_path[::-1]  # Reverse the path to start from the beginning
    astar_latent_path_latent = astar_latent_path_latent[::-1]
    # print("shortest path:", shortest_path, "\n")
    # print("shortest latent path:", shortest_path_latent, "\n")


# At this point, 'shortest_path' contains the path of latent points from start to end
       
    astar_lat_dist = np.zeros(len(astar_latent_path[1]))
    for i in range(len(astar_latent_path[1])):
        for j in range(len(astar_latent_path)-1):
            astar_lat_dist[i] += np.sqrt((astar_latent_path[j][i] - astar_latent_path[j+1][i])**2)
            # print("a\n")
    
                # # Plot the latent space in a scatter plot
    # plt.figure(figsize=(8, 6))
    # # plt.scatter(possible_lat[:, 0], possible_lat[:, 1], c='b', marker='o', s=10)
    # plt.scatter(possible_lat[:, 0], possible_lat[:, 1], c='b', marker='o', s=10)
    # plt.title('2D Latent Space')
    # plt.xlabel('Latent Dimension 1')
    # plt.ylabel('Latent Dimension 2')
    # plt.plot(shortest_path_latent[:, 0], shortest_path_latent[:, 1], c='g', marker='o', linestyle='-')
    # plt.plot(astar_latent_path_latent[:, 0], astar_latent_path_latent[:, 1], c='r', marker='o', linestyle='-')
    # plt.grid(True)
    # plt.show()
    
    astar_lat_dist_total = np.sum(astar_lat_dist)
    print("A star pathing\n")
    # print("pathing distances on each dimension")
    # print(astar_lat_dist)
    print("\n")
    print("total path distance")
    print(astar_lat_dist_total)
    print("\n")
    # print("average distance per step")
    # print(dist_total/len(trajectory))
    # print("\n")
    
    return trajectory, test_trajectory ,straight_trajectory, shortest_path, dist_total, test_dist_total, straight_dist_total, astar_dist_total, astar_lat_dist_total

# Define heuristic function (Euclidean distance)
# Define heuristic function (Euclidean distance)
def real_heuristic(node, end_node,conditional, mean, std, obj_type):
    # Calculate the estimated remaining cost in real space
    return calculate_real_distance(node, end_node,conditional, mean, std, obj_type)

def lat_heuristic(node, end_node,conditional):
    # Calculate the estimated remaining cost in real space
    return calculate_latent_distance(node, end_node,conditional)

def calculate_real_distance(candidate_point, current_latent_point,conditional, mean, std, obj_type):
    
    real_candidate = cvae_model.decode(candidate_point, conditional, mean, std, obj_type).detach()
    real_current = cvae_model.decode(current_latent_point, conditional, mean, std, obj_type).detach()
    
    # Calculate the squared differences in each dimension
    squared_diff = ((real_candidate- real_current) ** 2).sum()
    
    # Sum the squared differences and take the square root
    distance = np.sqrt((squared_diff))
    
    return distance.numpy()

def calculate_latent_distance(candidate_point, current_latent_point,conditional):

    
    # Calculate the squared differences in each dimension
    squared_diff = torch.sum((candidate_point - current_latent_point) ** 2)
    
    # Sum the squared differences and take the square root
    distance = torch.sqrt((squared_diff))
    
    return distance.numpy()

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
lat_dim = 2


# Load data
grasps, labels = load_data(args.data, robot='shadow')
pairs = np.load('data/final_pairs.npy')
pairs = pairs[:-12]
obj_used = []
    
print (f'Generating {len(pairs)} pairs!')
grasps_mean = np.mean(grasps, axis=0)
grasps_std = np.std(grasps, axis=0)
normalized_grasps = (grasps-grasps_mean)/grasps_std
mean_tensor = torch.from_numpy(grasps_mean)
std_tensor = torch.from_numpy(grasps_std)
test_object = torch.from_numpy(labels[:, 0])
test_object = test_object.unsqueeze(-1)


# creating different object groups depending on object type
unique_object_types = np.unique(labels[:,0])

grouped_arrays = {}
cluster_centers = {}

for obj_type in unique_object_types:
    indices = np.where(labels[:,0] == obj_type)[0]
    
    grouped_grasps = grasps[indices]
    
    cluster_center = np.mean(grouped_grasps, axis=0)
    
    grouped_arrays[obj_type] = grouped_grasps
    cluster_centers[obj_type] = cluster_center
    
    
pairwise_distances = {}

for obj_type, grouped_grasps in grouped_arrays.items():
    pairwise_dist = pdist(grouped_grasps)
    
    distance_matrix = squareform(pairwise_dist)
    
    pairwise_distances[obj_type] = distance_matrix
    
# Create an empty dictionary to store the pairwise distances between points and cluster centers
pairwise_distances_to_centers = {}

# Iterate through unique object types and their corresponding cluster centers
for obj_type, cluster_center in cluster_centers.items():
    # Calculate the pairwise distances between the points and the cluster center
    distances_to_center = cdist(grouped_arrays[obj_type], np.array([cluster_center]), 'euclidean')
    
    # Store the distances in the dictionary
    pairwise_distances_to_centers[obj_type] = distances_to_center

    
#trying a small value at the start, to have it sensible to similarities
sigma = 55

riemannian_matrices = {}

for obj_type in unique_object_types:
    distance_matrix = np.exp(-pairwise_distances_to_centers[obj_type]**2 / (2 * sigma**2))
    
    riemannian_matrices[obj_type] = distance_matrix
    
    
total_rows = sum(matrix.shape[0] for matrix in riemannian_matrices.values())

#a simple PCA encoding to test, since we need to encode the riemannian metric dictionary to match the size of grasps
#basic_pca = PCA(n_components=9)

result_array = np.empty((total_rows, 1), dtype=np.float64)

current_row = 0

for obj_type, matrix in riemannian_matrices.items():
    #have to use PCA to encode the metric dictionary, since it has 536x N size, 
    #where N depends on how many points of one object type there is
    encoded_matrix = matrix
    
    num_rows = encoded_matrix.shape[0]
    
    result_array [current_row : current_row + num_rows, :] = encoded_matrix
    
    current_row += num_rows
    

grasps, grasp_type, object_type, object_size = annotate_grasps(grasps, labels)

# indices_map = list(np.arange(grasps.shape[0])[idxs_n])

# To torch tensors
grasps = torch.from_numpy(grasps).float()
encoded_metrics = torch.from_numpy(result_array).float()
grasps_toread = grasps.detach().numpy()
object_size = torch.from_numpy(object_size).float().unsqueeze(-1)
object_type = torch.from_numpy(object_type).float()
# One hot encoding
object_type = one_hot(object_type.long(), class_size=3)
test_object = test_object.to(grasps.dtype)

y = torch.cat((object_type, object_size), dim=-1)

cvae_model = get_CVAE(args.model) 
recon_grasps, X_lat, _ = cvae_model(grasps, y, encoded_metrics, mean_tensor, std_tensor, test_object)

X_lat_toread = X_lat.detach().numpy()   

plt.figure(figsize=(8, 6))
plt.scatter(X_lat_toread[:, 0], X_lat_toread[:, 1], c='b', marker='o', s=10)
plt.title('2D Latent Space')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.grid(True)
plt.show()

inter_trajectories = list()
nrn_trajectories = list()
nln_trajectories = list()
astar_trajectories = list()

# pairs = np.load('data/final_pairs.npy')
# print (f'Generating {len(pairs)} pairs!')
# pairs = np.load('pairs_test.npy')
better_lat = 0
equal_lat = 0
worst_lat = 0
lat_dist = 0
real_dist = 0
straight_dist = 0
pair_index = 0
straight_dist = 0
astar_dist = 0
astar_lat_dist = 0
for p in pairs:
    nrn_traj, nln_traj, inter_traj, astar_traj, total_dist, test_total_dist, straight_total_dist, astar_total_dist, astar_lat_total_dist = get_trajectory(p[0], p[1])
    lat_dist += test_total_dist
    real_dist += total_dist
    straight_dist += straight_total_dist
    astar_dist += astar_total_dist
    astar_lat_dist += astar_lat_total_dist
    if test_total_dist < total_dist:
        better_lat += 1
    elif test_total_dist > total_dist:
        worst_lat += 1
    else:
        equal_lat +=1
    
    # trajectories.append((traj,total_dist))
    inter_trajectories.append(inter_traj)
    nrn_trajectories.append(nrn_traj)
    nln_trajectories.append(nln_traj)
    astar_trajectories.append(astar_traj)
    
print("latent neighbors was better in ", better_lat, " cases\n")
print("latent neighbors was equal in ", equal_lat, " cases\n")
print("latent neighbors was worst in ", worst_lat, " cases\n")
print("real neighbors distance was ", real_dist, "\n")
print("latent neighbors distance was ", lat_dist, "\n")
print("difference: ", real_dist - lat_dist, "\n")

print("astar real distance: ", astar_dist, "\n")
print("astar lat distance: ", astar_lat_dist, "\n")
print("straight line distance: ", straight_dist, "\n")


grasp_idxs = np.asarray(pairs)
model_name = args.model.split('/')[-1]
save_trajectories(grasp_idxs, nrn_trajectories, nln_trajectories, astar_trajectories, inter_trajectories, model_name, n_steps,'')

