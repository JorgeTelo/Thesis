import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from data_loader import load_data, annotate_grasps
from utils import vae_loss
from utils import one_hot
from cvae import CVAE
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser(description='Train AE for ICUB')
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
    for batch_idx, (data, label, riemannian_metric) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, label, riemannian_metric)
        loss = vae_loss(recon_batch, data, mu, logvar, kl_weight)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    return train_loss

args = parse_args()

torch.manual_seed(args.seed)
kl_weight = args.kl

grasps, labels = load_data(args.data, robot='icub')

# creating different object groups depending on object type
unique_object_types = np.unique(labels[:,0])

grouped_arrays = {}

for obj_type in unique_object_types:
    indices = np.where(labels[:,0] == obj_type)[0]
    
    grouped_grasps = grasps[indices]
    
    grouped_arrays[obj_type] = grouped_grasps
    
pairwise_distances = {}

for obj_type, grouped_grasps in grouped_arrays.items():
    pairwise_dist = pdist(grouped_grasps)
    
    distance_matrix = squareform(pairwise_dist)
    
    pairwise_distances[obj_type] = distance_matrix
    
#trying a small value at the start, to have it sensible to similarities
sigma = 75

riemannian_matrices = {}

for obj_type, dist in pairwise_distances.items():
    distance_matrix = np.exp(-dist**2 / (2 * sigma**2))
    
    riemannian_matrices[obj_type] = distance_matrix
    
    
total_rows = sum(matrix.shape[0] for matrix in riemannian_matrices.values())

#a simple PCA encoding to test, since we need to encode the riemannian metric dictionary to match the size of grasps
basic_pca = PCA(n_components=9)

result_array = np.empty((total_rows, 9), dtype=np.float64)

current_row = 0

for obj_type, matrix in riemannian_matrices.items():
    #have to use PCA to encode the metric dictionary, since it has 536x N size, 
    #where N depends on how many points of one object type there is
    encoded_matrix = basic_pca.fit_transform(matrix)
    
    num_rows = encoded_matrix.shape[0]
    
    result_array [current_row : current_row + num_rows, :] = encoded_matrix
    
    current_row += num_rows
    
    
grasps, grasp_type, object_type, object_size = annotate_grasps(grasps, labels)

## To torch tensors
## Data
X = torch.from_numpy(grasps).float()
## Metrics
encoded_metrics = torch.from_numpy(result_array).float()
## Labels
object_size = torch.from_numpy(object_size).float().unsqueeze(-1)
object_type = torch.from_numpy(object_type).float()
object_type = one_hot(object_type.long(), class_size=3)
y = torch.cat((object_type, object_size), dim=-1)

input_dim = 9
hidden_dim = 64
latent_dim = 2
conditional_dim = 4
metric_dim = 9 #number of PCA components

train_data = torch.utils.data.TensorDataset(X, y, encoded_metrics)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1)

model = CVAE(in_dim = input_dim, hid_dim = hidden_dim, lat_dim = latent_dim, c_dim=conditional_dim, metric_dim = metric_dim)
best_model = CVAE(in_dim = input_dim, hid_dim = hidden_dim, lat_dim = latent_dim, c_dim=conditional_dim, metric_dim = metric_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

cur_best = None
train_losses = []
for epoch in tqdm(range(1, args.epochs + 1), desc='Epochs'):
    train_losses.append(train(epoch))
    recon_data, _, _ = model(X, y, encoded_metrics)
    mse = F.mse_loss(recon_data, X).item()

    is_best = not cur_best or mse < cur_best
    if is_best:
        cur_best = mse
        best_model.load_state_dict(model.state_dict())

filename = 'output/cvae'
filename += '_seed_' + str(args.seed) 
filename += '_epochs_' + str(args.epochs) 
filename += '_hiddim_' + str(hidden_dim)
filename += '_kl_' + str(args.kl)

recon_data, _, _ = model(X, y, encoded_metrics)
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
