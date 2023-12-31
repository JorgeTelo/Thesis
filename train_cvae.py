import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from scipy.spatial.distance import pdist, squareform, cdist

import matplotlib.pyplot as plt


from data_loader import load_data, annotate_grasps
from utils import vae_loss
from utils import riemannian_loss
from utils import one_hot
from utils import compute_geodesics_full, compute_pairwise_geodesics, compute_geodesics_euclidean, compute_geodesics_center
from cvae import CVAE

import itertools

def parse_args():
    parser = argparse.ArgumentParser(description='Train AE for Shadowhand')
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

def train(epoch):
    model.train()
    train_loss = 0
    
    points_encoded = []
    
    for batch_idx_dummy, (datadummy, labeldummy, riemannian_metricdummy, dmean, dstd, dobj_type) in enumerate(train_loader):
            _, mdummy, _ = model(datadummy, labeldummy, riemannian_metricdummy, dmean, dstd, dobj_type)
            points_encoded.append(mdummy)
    
    points_encoded_tensor = torch.stack(points_encoded)
    geodesic_distances = compute_geodesics_full(points_encoded_tensor, G_matrix) 
    # print("\n", geodesic_distances, "\n")   
            
    latent_center = torch.mean(points_encoded_tensor, dim = 0)
    cluster_centers_latent = {}
    geodesic_center_distances_list = []
    dist_latent_center = []
    previous_idx = 0
    for obj_type, grouped_grasps in grouped_arrays.items():
        current_grasps = []
        num_points = len(grouped_grasps)
        for i in range(num_points):
            current_grasps.append(points_encoded[i+previous_idx])
            dist_center = torch.sqrt(torch.sum((points_encoded[i+previous_idx]-latent_center)**2))
            dist_latent_center.append(dist_center)
            count_iter = i
        
        previous_idx = count_iter
        
        stacked_grasps = torch.stack(current_grasps)
        cluster_center_latent = torch.mean(stacked_grasps, dim = 0)
        for j in range(num_points):
            geodesic_dist_center = compute_pairwise_geodesics(stacked_grasps[j], cluster_center_latent, G_matrix)
            geodesic_center_distances_list.append(geodesic_dist_center)

    dist_latent_center_tensor = torch.stack(dist_latent_center)
    dist_latent_center_tensor = dist_latent_center_tensor.unsqueeze(-1)
    #geodesic_center_distances = compute_geodesics_center(dist_latent_center_tensor, G_matrix)
    geodesic_center_distances = torch.stack(geodesic_center_distances_list)
        #print("a\n")    
    
    # for cluster, tensor in cluster_centers_latent.items():
    #     point = tensor.detach().numpy()
    #     plt.scatter(point[0][0], point[0][1], c='g', marker='o', s=10)
        
    # own_cluster_distance = {}
    # other_cluster_distance = {}
        
    # previous_idx = 0
    # for obj_type, grouped_grasps in grouped_arrays.items():
    #     same_distance = []
    #     other_distance = []
    #     num_points = len(grouped_grasps)
    #     for i in range(num_points):
    #         # print("current encoded point: ", points_encoded[i+previous_idx], "\n")
    #         # print("current cluster center", cluster_centers_latent[obj_type], "\n")
    #         same_distance.append(torch.sqrt(torch.sum((points_encoded[i+previous_idx] - cluster_centers_latent[obj_type])**2)))
    #         for obj_type2 in unique_object_types:
    #             if obj_type2 != obj_type:
    #                 # print("current encoded point: ", points_encoded[i+previous_idx], "\n")
    #                 # print("current cluster center", cluster_centers_latent[obj_type], "\n")
    #                 other_distance.append(torch.sqrt(torch.sum((points_encoded[i+previous_idx] - cluster_centers_latent[obj_type2])**2)))
    #         count_iter = i
        
        
    #     own_cluster_distance[obj_type] = torch.stack(same_distance)
    #     other_cluster_distance[obj_type] = torch.stack(other_distance)
    #     previous_idx = count_iter
        
        
    # geodesic_distances = compute_geodesics_full(same_distance, G_matrix) 
    # print("\n", geodesic_distances, "\n")    

    #regularization term, to make sure clusters are separated
    weight_first_loss = 1.0
    weight_own = 5.0 # positive means training will prio making this smaller
    weight_other = -1.0 
    weight_cluster = -10.0 #adjust as needed
    reg_term_cluster = 0
    
    min_cluster_distance = 20

    
    for batch_idx, (data, label, riemannian_metric, mean, std, obj_num) in enumerate(train_loader):
        #print("current data: ", data, "\n")
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, label, riemannian_metric, mean,std, obj_num)
        first_loss = riemannian_loss(recon_batch, data, mu, logvar, kl_weight, 
                                     geodesic_distances[batch_idx], pairwise_distances_tensor[batch_idx], 
                                     geodesic_center_distances[batch_idx], riemannian_array[batch_idx], sigma)
        # first_loss = vae_loss(recon_batch, data, mu, logvar, kl_weight)
        reg_term_own = 0
        reg_term_other = 0  

#####################################
        #add the cluster separation term
        
        # for obj_type1, obj_type2 in itertools.combinations(unique_object_types,2):
        #     # Term 1: Encourage points to stay close to their own cluster center
        #     reg_term_own += (torch.mean(own_cluster_distance[obj_type1])).item()
            

        #     # Term 2: Discourage points from being close to other cluster centers
        #     reg_term_other += (torch.mean(other_cluster_distance[obj_type1])).item()
        #     # Term 3: Encourage cluster centers to be far away from each other
        #     # differences = [value - min_cluster_distance for value in pairwise_latent_dist_centers.values()]
        #     # reg_term_cluster = torch.mean(torch.tensor(differences))
            
        # total_reg = weight_own*reg_term_own + (weight_other*reg_term_other) #+ (weight_cluster*reg_term_cluster)
        # print("Reg term cluster: ", reg_term_cluster, "\n")
        # print("Reg term own: ", reg_term_own, "\n")
        # print("Reg term other: ", reg_term_other, "\n")    
        # print("Total reg: ", total_reg, "\n")
        # print("First loss: ", first_loss, "\n")    
        loss = weight_first_loss*first_loss #+ total_reg
        # print("current batch", batch_idx, "\n")
        
        loss.backward(retain_graph = True)
        train_loss += loss.item()
        optimizer.step()
        # print("\nFull loss: ", loss, "\n")
    
    train_loss /= len(train_loader.dataset)
    print("\nTraining loss: ", train_loss, "\n")
    # print("Reg term cluster: ", reg_term_cluster, "\n")
    # print("Reg term own: ", reg_term_own, "\n")
    # print("Reg term other: ", reg_term_other, "\n")    
    # print("Total reg: ", total_reg, "\n")
    # print("First loss: ", first_loss, "\n")    
    # print("Total loss: ", loss, "\n")
    
    print("Getting out of train loop\n")
    return train_loss


def kl_weight_annealing(epoch, max_epochs):
    start_weight = 15
    end_weight = 2.5
    annealing_epochs = 0.7*max_epochs
    if epoch < annealing_epochs:
        return start_weight + (end_weight - start_weight)*(epoch/annealing_epochs)
    else:
        return end_weight
    
    
# METRIC 2
# def define_riemannian_metric(unique_object_types, pairwise_distances):
# # Initialize the Riemannian metric matrix G as an identity matrix.
#     G = np.eye(latent_dim)
    
#     # Define a scaling factor to adjust the metric.
#     scaling_factor1 = 0.1  # You can adjust this value based on your problem.
#     scaling_factor2 = 0.0011
#     avg_distance = 0
#     # Objective 1: Encourage Clustering
#     for obj_type in unique_object_types:
#         # Calculate the average pairwise distance between points of the same object type in the real space.
#         avg_distance += np.mean(pairwise_distances[obj_type])
    
#         # Set the corresponding diagonal element in G to encourage clustering.
#         for i in range(latent_dim):
#             G[i,i] += scaling_factor1 * avg_distance/(len(unique_object_types))
#         # G[0, 0] += scaling_factor1 * avg_distance/(len(unique_object_types))
#         # G[1, 1] += scaling_factor1 * avg_distance/(len(unique_object_types))
    
#     avg_min_dist = 0
#     # Objective 2: Discourage Mixing
#     for obj_type1 in unique_object_types:
#         for obj_type2 in unique_object_types:
#               if obj_type1 != obj_type2:
#                 # Calculate the minimum pairwise distance between points of different object types in the real space.
#                 min_distance = float('inf')
#                 for i in range(len(grouped_arrays[obj_type1])):
#                     for j in range(len(grouped_arrays[obj_type2])):
#                         distance_ij = np.sqrt(np.sum((grouped_arrays[obj_type1][i]-grouped_arrays[obj_type2][j])**2))
#                         # print("current_dist: ",distance_ij,"\n")
#                         if distance_ij < min_distance:
#                             min_distance = distance_ij
#                             # print("new min: ", min_distance, "\n")
#                 # print("min dist between obj_type ", obj_type1, "and obj_type ", obj_type2, " : ", min_distance, "\n")
#                 avg_min_dist += min_distance
    
#                 # Set the off-diagonal elements in G to discourage mixing.
#                 for a in range(latent_dim):
#                     for b in range(latent_dim):
#                         if a != b:
#                             G[a,b] += scaling_factor2 * avg_min_dist/(len(unique_object_types))
    
#     return G


# ## METRIC 3
# def define_riemannian_metric(unique_object_types, pairwise_distances):
# # Initialize the Riemannian metric matrix G as an identity matrix.
#     G = np.eye(latent_dim)
    
#     # Define a scaling factor to adjust the metric.
#     scaling_factor1 = 0.00115  # You can adjust this value based on your problem.
#     scaling_factor2 = 1
#     avg_distance = 0
#     # Objective 1: Encourage Clustering
#     for i in range(len(full_dist_matrix)):
#         # Calculate the average pairwise distance between points in the real space
#         avg_distance += np.mean(full_dist_matrix[i])
    
#         # Set the corresponding diagonal element in G to encourage clustering.
#         for j in range(latent_dim):
#             G[j,j] += scaling_factor1 * avg_distance/(len(unique_object_types))
#         # G[0, 0] += scaling_factor1 * avg_distance/(len(full_dist_matrix)-1)
#         # G[1, 1] += scaling_factor1 * avg_distance/(len(full_dist_matrix)-1)
    
#     avg_min_dist = 0
#     # for i in range(len(full_dist_matrix)):
#     #     full_dist_matrix[i][i] = 50000
        
#     # min_distances = full_dist_matrix.min(axis=1)
#     # avg_min_dist = np.sum(min_distances)
    
#     max_distance_indices = np.unravel_index(np.argmax(full_dist_matrix), full_dist_matrix.shape)
    
#     # # Set the off-diagonal elements in G to discourage mixing.
#     # G[0, 1] += scaling_factor2 * avg_min_dist/(len(full_dist_matrix))
#     # G[1, 0] += scaling_factor2 * avg_min_dist/(len(full_dist_matrix))
    
#     # G[0, 1] = scaling_factor2 * full_dist_matrix[max_distance_indices]
#     # G[1, 0] = scaling_factor2 * full_dist_matrix[max_distance_indices]
#     for a in range(latent_dim):
#         for b in range(latent_dim):
#             if a != b:
#                 G[a,b] += scaling_factor2 * full_dist_matrix[max_distance_indices]
    
#     return G

#metric 1
def define_riemannian_metric(unique_object_types, pairwise_distances_to_own_center):
# Initialize the Riemannian metric matrix G as an identity matrix.
    G = np.eye(latent_dim)
    
    # Define a scaling factor to adjust the metric.
    scaling_factor1 = 1.1  # You can adjust this value based on your problem.
    scaling_factor2 = 0.01
    avg_distance = 0
    # Objective 1: Encourage Clustering
    for obj_type in unique_object_types:
        # Calculate the average pairwise distance between points of the same object type in the real space.
        avg_distance += np.mean(pairwise_distances[obj_type])
    
        # Set the corresponding diagonal element in G to encourage clustering.
        for i in range(latent_dim):
            G[i,i] += scaling_factor1 * avg_distance/(len(unique_object_types))
        # G[0, 0] += scaling_factor1 * avg_distance/(len(unique_object_types))
        # G[1, 1] += scaling_factor1 * avg_distance/(len(unique_object_types))
        # G[2, 2] += scaling_factor1 * avg_distance/(len(unique_object_types))
    
    avg_min_dist = 0
    # Objective 2: Discourage Mixing
    for obj_type1 in unique_object_types:
        for obj_type2 in unique_object_types:
              if obj_type1 != obj_type2:
                # Calculate the minimum pairwise distance between points of different object types in the real space.
                min_distance = float('inf')
                for i in range(len(grouped_arrays[obj_type1])):
                    for j in range(len(grouped_arrays[obj_type2])):
                        distance_ij = np.sqrt(np.sum((grouped_arrays[obj_type1][i]-grouped_arrays[obj_type2][j])**2))
                        # print("current_dist: ",distance_ij,"\n")
                        if distance_ij < min_distance:
                            min_distance = distance_ij
                            # print("new min: ", min_distance, "\n")
                # print("min dist between obj_type ", obj_type1, "and obj_type ", obj_type2, " : ", min_distance, "\n")
                avg_min_dist += min_distance
    
                # Set the off-diagonal elements in G to discourage mixing.
                for i in range(latent_dim):
                    for j in range(latent_dim):
                        if i != j:
                            G[i,j] += scaling_factor2 * avg_min_dist/(len(unique_object_types))
                # G[0, 1] += scaling_factor2 * avg_min_dist/(len(unique_object_types))
                # G[0, 2] += scaling_factor2 * avg_min_dist/(len(unique_object_types))
                # G[1, 0] += scaling_factor2 * avg_min_dist/(len(unique_object_types))
                # G[2, 0] += scaling_factor2 * avg_min_dist/(len(unique_object_types))
                # G[1, 2] += scaling_factor2 * avg_min_dist/(len(unique_object_types))
                # G[2, 1] += scaling_factor2 * avg_min_dist/(len(unique_object_types))
    
    return G

    
args = parse_args()
latent_dim = 2
torch.autograd.set_detect_anomaly(False)

torch.manual_seed(args.seed)
kl_weight = args.kl

#labels[0] = object_type, labels[1] = grasp
grasps, labels = load_data(args.data, robot='shadow')
grasps_mean = np.mean(grasps, axis=0)
grasps_std = np.std(grasps, axis=0)
normalized_grasps = (grasps-grasps_mean)/grasps_std
mean_tensor = torch.from_numpy(grasps_mean)
std_tensor = torch.from_numpy(grasps_std)

# creating different object groups depending on object type
unique_object_types = np.unique(labels[:,0])

grouped_arrays = {}

for obj_type in unique_object_types:
    indices = np.where(labels[:,0] == obj_type)[0]
    
    grouped_grasps = grasps[indices]
       
    grouped_arrays[obj_type] = grouped_grasps
    
#Distnaces between points inside 1 object tyoe
pairwise_distances = {}

for obj_type, grouped_grasps in grouped_arrays.items():
    pairwise_dist = pdist(grouped_grasps)
    
    distance_matrix = squareform(pairwise_dist)
    
    pairwise_distances[obj_type] = distance_matrix


#complete tensor with all pairwise_distance 536x536
full_pairwise_dist = pdist(grasps)
full_dist_matrix = squareform(full_pairwise_dist)
        
pairwise_distances_tensor = torch.from_numpy(full_dist_matrix)

#Neighbor pairing in real space, inside each object type    
previous_idx = 0
neighbor_real_pairs = {}
for obj_type, distances in pairwise_distances.items():
    pairs = []
    for i in range(len(distances)):
        min_distance = float('inf')
        neighbor_index = -1
        for j in range(len(distances)):
            if i != j and distances[i, j] < min_distance:
                min_distance = distances[i, j]
                neighbor_index = j
            # Add the pair of indices to the list
        pairs.append([i+previous_idx, neighbor_index+previous_idx])
        count_iter = i
    previous_idx += count_iter+1
    # Store the list of pairs in the dictionary
    neighbor_real_pairs[obj_type] = pairs


# Create an empty dictionary to store the pairwise distances between points and own cluster centers
#careful with this, since this is only used for the riemannian metric, we use non normalized data
pairwise_distances_to_own_centers = {}
nonnorm_grouped_arrays = {}
cluster_centers = {}

for obj_type in unique_object_types:
    indices = np.where(labels[:,0] == obj_type)[0]
    
    nonnorm_grouped_grasps = grasps[indices]
    
    cluster_center = np.mean(nonnorm_grouped_grasps, axis=0)
    
    nonnorm_grouped_arrays[obj_type] = nonnorm_grouped_grasps
    cluster_centers[obj_type] = cluster_center
# Iterate through unique object types and their corresponding cluster centers
for obj_type, cluster_center in cluster_centers.items():
    # Calculate the pairwise distances between the points and the cluster center
    distances_to_center = cdist(grouped_arrays[obj_type], np.array([cluster_center]), 'euclidean')
    
    # Store the distances in the dictionary
    pairwise_distances_to_own_centers[obj_type] = distances_to_center
    
    
pairwise_distance_center = []
previous_idx = 0
for obj_type, grasp in grouped_arrays.items():
    distance = []
    for i in range(len(grasp)):
        distance = np.sqrt(np.sum((grasp[i]-grasps_mean)**2))
        pairwise_distance_center.append(distance)

pairwise_dist_center = np.array(pairwise_distance_center)
    
##metric 2 and 3
# G_matrix = define_riemannian_metric(unique_object_types, pairwise_distances)

#matrix 1
G_matrix = define_riemannian_metric(unique_object_types, pairwise_distances_to_own_centers)
#trying a small value at the start, to have it sensible to similarities
sigma = 25

riemannian_matrices = {}

for obj_type in unique_object_types:
    distance_matrix = np.exp(-pairwise_distances_to_own_centers[obj_type]**2 / (2 * sigma**2))
    
    riemannian_matrices[obj_type] = distance_matrix
    
    
total_rows = sum(matrix.shape[0] for matrix in riemannian_matrices.values())


#a simple PCA encoding to test, since we need to encode the riemannian metric dictionary to match the size of grasps
#basic_pca = PCA(n_components=9)

# riemannian_array = np.empty((total_rows, 1), dtype=np.float64)

# current_row = 0

# for obj_type, matrix in riemannian_matrices.items():
#     #have to use PCA to encode the metric dictionary, since it has 536x N size, 
#     #where N depends on how many points of one object type there is
#     encoded_matrix = matrix
    
#     num_rows = encoded_matrix.shape[0]
    
#     riemannian_array [current_row : current_row + num_rows, :] = encoded_matrix
    
#     current_row += num_rows
    
riemannian_array = []
result_array = []
for i in range(len(pairwise_dist_center)):
    metric = np.exp(-pairwise_dist_center[i]**2 / (2 * sigma**2))
    riemannian_array.append(metric)
    
result_array = riemannian_array
riemannian_array = np.array(riemannian_array)
# cluster_centers_latent = {}
# grid_size = 20
# grid_spacing = grid_size / np.sqrt(len(unique_object_types))

# for i in range(1,len(unique_object_types)+1):
#     x = (i % np.sqrt(len(unique_object_types))) * grid_spacing -10
#     y = (i // np.sqrt(len(unique_object_types))) * grid_spacing -10
#     center = torch.tensor([x,y])
#     cluster_centers_latent[i] = center
  
grasps, grasp_type, object_type, object_size = annotate_grasps(grasps, labels)

## To torch tensors
## Data
X = torch.from_numpy(grasps).float()
## Metrics
encoded_metrics = torch.from_numpy(riemannian_array).float()
riemannian_array = torch.from_numpy(riemannian_array).float()
## Labels
object_size = torch.from_numpy(object_size).float().unsqueeze(-1)
object_type = torch.from_numpy(object_type).float()
object_type = one_hot(object_type.long(), class_size=3)
y = torch.cat((object_type, object_size), dim=-1)
y_toread = y.detach().numpy()
grasps_mean = torch.from_numpy(grasps_mean)
grasps_mean = grasps_mean.unsqueeze(0)
grasps_mean = grasps_mean.repeat(len(grasps), 1)
grasps_std = torch.from_numpy(grasps_std)
grasps_std = grasps_std.unsqueeze(0)
grasps_std = grasps_std.repeat(len(grasps), 1)
test_object = torch.from_numpy(labels[:, 0])
test_object = test_object.unsqueeze(-1)
test_object = test_object.to(X.dtype)



input_dim = len(grasps[1])
hidden_dim = 128
conditional_dim = 4
metric_dim = 1 #number of PCA components

train_data = torch.utils.data.TensorDataset(X, y, encoded_metrics, grasps_mean, grasps_std, test_object)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1)

model = CVAE(in_dim = input_dim, hid_dim = hidden_dim, lat_dim = latent_dim, c_dim=conditional_dim, metric_dim = metric_dim)
best_model = CVAE(in_dim = input_dim, hid_dim = hidden_dim, lat_dim = latent_dim, c_dim=conditional_dim, metric_dim = metric_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

cur_best = None
train_losses = []
mse_list = []
for epoch in tqdm(range(1, args.epochs + 1), desc='Epochs'):
    # kl_weight = kl_weight_annealing(epoch, args.epochs)
    kl_weight = 1
    train_losses.append(train(epoch))
    recon_data, X_lat, _ = model(X, y, encoded_metrics, grasps_mean, grasps_std, test_object)
    X_lat_toread = X_lat.detach().numpy()
    mse = F.mse_loss(recon_data, X).item()
    print("Epoch: ", epoch, "\n")
    print("MSE: ", mse, "\n")
    mse_list.append(mse)
    

    is_best = not cur_best or mse < cur_best
    if is_best:
        cur_best = mse
        best_model.load_state_dict(model.state_dict())
        plt.figure(figsize=(8, 6))
        plt.scatter(X_lat_toread[:, 0], X_lat_toread[:, 1], c='b', marker='o', s=10)
        plt.title('2D Latent Space')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.grid(True)
        plt.show()
        


filename = 'output/cvae'
filename += '_seed_' + str(args.seed) 
filename += '_epochs_' + str(args.epochs) 
filename += '_hiddim_' + str(hidden_dim)
filename += '_kl_' + str(args.kl)

recon_data, X_lat, _ = best_model(X, y, encoded_metrics, grasps_mean, grasps_std, test_object)

X_lat_toread = X_lat.detach().numpy()
plt.figure(figsize=(8, 6))
plt.scatter(X_lat_toread[:, 0], X_lat_toread[:, 1], c='b', marker='o', s=10)
plt.title('Best 2D Latent Space')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.grid(True)
plt.show()


#checking resulting latent neighboring pairs  
grouped_latent_arrays = {}
for obj_type in unique_object_types:
    indices = np.where(labels[:,0] == obj_type)[0]
    
    grouped_latent_grasps = X_lat_toread[indices]
        
    grouped_latent_arrays[obj_type] = grouped_latent_grasps
#complete tensor with all pairwise_distance 536x536
pairwise_latent_distances = {}
for obj_type, grouped_latent_grasps in grouped_latent_arrays.items():
    pairwise_latent_dist = pdist(grouped_latent_grasps)
    
    latent_distance_matrix = squareform(pairwise_latent_dist)
    
    pairwise_latent_distances[obj_type] = latent_distance_matrix


#Neighbor pairing in latent space, inside each object type    
previous_idx = 0
neighbor_latent_pairs = {}
for obj_type, distances in pairwise_latent_distances.items():
    pairs = []
    for i in range(len(distances)):
        min_distance = float('inf')
        neighbor_index = -1
        for j in range(len(distances)):
            if i != j and distances[i, j] < min_distance:
                min_distance = distances[i, j]
                neighbor_index = j
            # Add the pair of indices to the list
        pairs.append([i+previous_idx, neighbor_index+previous_idx])
        count_iter = i
    previous_idx += count_iter+1
    # Store the list of pairs in the dictionary
    neighbor_latent_pairs[obj_type] = pairs


mse = F.mse_loss(recon_data, X).item()
print ("CVAE reconstruction error: ", mse)

torch.save({
    'state_dict': best_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'layers': [input_dim, hidden_dim, latent_dim, conditional_dim, metric_dim],
    'mse': mse,
    },  filename)

plt.plot(train_losses, label='CVAE', c='b')
plt.legend()
plt.title('Train loss for CVAE with KL weight: ' + str(kl_weight))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


