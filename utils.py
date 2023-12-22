import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from cvae import CVAE
from scipy.spatial.distance import pdist, squareform, cdist

output_dir = 'output/'

grasp_type_names = ['Tripod',
                    'Palmar Pinch',
                    'Lateral',
                    'Writing Tripod',
                    'Parallel Extension',
                    'Adduction Grip',
                    'Tip Pinch',
                    'Lateral Tripod']

object_configuration_names = ["Big Green Ball",
                              "Medium Blue Ball",
                              "Small White Ball",
                              "Big Red Cylinder Top",
                              "Big Red Cylinder Side",
                              "Medium Blue Cylinder Top",
                              "Medium Blue Cylinder Side",
                              "Small Red Cylinder Top",
                              "Small Red Cylinder Side",
                              "Pen",
                              "Small Purple Cube",
                              "Blue Box Large Side",
                              "Blue Box Small Side",
                              "Orange Box Large Side",
                              "Orange Box Small Side",
                              "Red Box Large Side",
                              "Red Box Small Side",
                              "Red Box Medium Side",
                              "Yellow Box Small Side",
                              "Yellow Box Large Side"]

def one_hot(classes, class_size=8): 
    targets = torch.zeros(classes.size(0), class_size)
    for i, label in enumerate(classes):
        targets[i, label-1] = 1
    return Variable(targets)

def reconstruction_loss (recon_x, x):
    return F.mse_loss(recon_x, x)

def kl_divergence_loss (mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss (recon_x, x, mu, logvar, kl_weight):
    BCE = reconstruction_loss(recon_x, x)
    KLD = kl_divergence_loss(mu, logvar)
    return BCE + kl_weight * KLD

def riemannian_loss(recon_x, x, mu, logvar, kl_weight, geodesic_distances, pairwise_distances, geodesic_center, riemannian_array, sigma):
    BCE = reconstruction_loss(recon_x, x)
    KLD = kl_weight*kl_divergence_loss(mu, logvar)
    # print("BCE: ", BCE, "\n")
    # print("KLD: ", KLD, "\n")
       
    # Define a loss term to encourage neighbors in real space to be neighbors in latent space
    pairwise_distance_loss = compute_pairwise_distance_loss(geodesic_distances, pairwise_distances)
    center_distance_loss = compute_center_distance_loss(geodesic_center, riemannian_array, sigma)
    # print(pairwise_distance_loss,"\n")
    return BCE + KLD + pairwise_distance_loss #+ center_distance_loss

def compute_geodesics_full(mu, G):
    
    # mu is the matrix of latent space points (N x latent_dim)
    # G is the Riemannian metric matrix (latent_dim x latent_dim)
    # Compute the Mahalanobis distance using the Riemannian metric
    # Mahalanobis distance squared = (x - y)^T * G * (x - y)
    # Where x and y are latent space points
    G_tensor = torch.from_numpy(G)
    G_tensor = G_tensor.to(mu.dtype)
    # Expand mu for pairwise differences
    mu_expand = mu.unsqueeze(2)  # Shape: (N, 1, latent_dim)
    mu_diff = mu_expand - mu  # Pairwise differences, Shape: (N, N, latent_dim)

    # Compute the geodesic distances using the Riemannian metric G
    # D^2(x, y) = (x - y)^T * G * (x - y)
    geodesic_distances = torch.sqrt(torch.einsum('ijkl,kl,ijkl->ij', mu_diff, G_tensor, mu_diff))

    return geodesic_distances

def compute_geodesics_center(mu, G):
    
    # mu is the matrix of latent space points (N x latent_dim)
    # G is the Riemannian metric matrix (latent_dim x latent_dim)
    # Compute the Mahalanobis distance using the Riemannian metric
    # Mahalanobis distance squared = (x - y)^T * G * (x - y)
    # Where x and y are latent space points
    G_tensor = torch.from_numpy(G)
    G_tensor = G_tensor.to(mu.dtype)
    # Expand mu for pairwise differences
    mu_expand = mu.unsqueeze(2)  # Shape: (N, 1, latent_dim)
    mu_diff = mu_expand - mu  # Pairwise differences, Shape: (N, N, latent_dim)

    # Compute the geodesic distances using the Riemannian metric G
    # D^2(x, y) = (x - y)^T * G * (x - y)
    geodesic_distances = torch.sqrt(torch.einsum('ijl,kl,ijl->ij', mu_diff, G_tensor, mu_diff))

    return geodesic_distances

def compute_geodesics_euclidean(mu):
    
    # mu is the matrix of latent space points (N x latent_dim)
    # G is the Riemannian metric matrix (latent_dim x latent_dim)
    # Compute the Euclidean distance
    # Expand mu for pairwise differences
    mu = mu.detach().numpy()
    pairwise_dist = pdist(mu)
    
    geodesic_distances = squareform(pairwise_dist)

    return geodesic_distances


def compute_pairwise_geodesics(mu1, mu2, G):
    # mu1 and mu2 are individual latent space points (1x2 tensors)
    # G is the Riemannian metric matrix (2x2)
    G_tensor = torch.from_numpy(G)
    G_tensor = G_tensor.to(mu1.dtype)
    # Compute the Mahalanobis distance using the Riemannian metric
    # Mahalanobis distance squared = (x - y)^T * G * (x - y)
    # Where x and y are latent space points

    # Compute the geodesic distance between mu1 and mu2 using the Riemannian metric G
    mu_diff = mu1 - mu2
    geodesic_distance = torch.matmul(mu_diff.unsqueeze(0), torch.matmul(G_tensor, mu_diff.unsqueeze(-1)))

    return geodesic_distance


def compute_pairwise_distance_loss(geodesic_distances, real_space_distances):
    real_space_distances = real_space_distances.detach().numpy()
    geodesic_distances = geodesic_distances.detach().numpy()
    # Compute the pairwise distance loss between geodesic distances (latent space) and real space distances
    # geodesic_distances: Tensor of shape (N, N) containing geodesic distances in latent space
    # real_space_distances: Tensor of shape (N, N) containing pairwise distances in real space

    # Compute the difference between real space distances and geodesic distances
    difference = np.sqrt(np.sum((real_space_distances - geodesic_distances)**2))

    # Sum up the squared differences to get the pairwise distance loss
    loss = difference#/len(geodesic_distances)

    return loss

def compute_center_distance_loss(geodesic_distances, real_space_distances, sigma):
    real_space_distances = real_space_distances.detach().numpy()
    geodesic_distances = geodesic_distances.detach().numpy()
    geodesic_distances = np.exp(-geodesic_distances**2 / (2 * sigma**2))
    # Compute the pairwise distance loss between geodesic distances (latent space) and real space distances
    # geodesic_distances: Tensor of shape (N, N) containing geodesic distances in latent space
    # real_space_distances: Tensor of shape (N, N) containing pairwise distances in real space

    # Compute the difference between real space distances and geodesic distances
    difference = np.sqrt(np.sum((real_space_distances - geodesic_distances)**2))

    # Sum up the squared differences to get the pairwise distance loss
    loss = difference/len(geodesic_distances)

    return loss

def get_lininter(p1, p2, t):
    return (1 - t) * p1 + t * p2

# def get_CVAE(vae_file):
#     print (vae_file)
#     # assert exists(vae_file), "No trained VAE in the logdir..."
#     state = torch.load(vae_file)
#     layers = state['layers']
#     vae = CVAE(in_dim = layers[0], hid_dim = layers[1], lat_dim = layers[2],
#             c_dim = 4)
#     vae.load_state_dict(state['state_dict'])
#     return vae

def get_VAE(vae_file):
    state = torch.load(vae_file)
    layers = state['layers']
    vae = VAE(in_dim = layers[0], hid_dim = layers[1], lat_dim = layers[2])
    vae.load_state_dict(state['state_dict'])
    return vae

def get_AE(ae_file):
    state = torch.load(ae_file)
    layers = state['layers']
    ae = AE(in_dim = layers[0], hid_dim = layers[1], lat_dim = layers[2])
    ae.load_state_dict(state['state_dict'])
    return ae

def get_CVAE(cvae_file):
    filename = 'output/'
    filename+= str(cvae_file)
    state = torch.load(filename)
    layers = state['layers']
    cvae = CVAE(in_dim = layers[0], hid_dim = layers[1], lat_dim = layers[2],
            c_dim = layers[3], metric_dim = layers[4])
    cvae.load_state_dict(state['state_dict'])
    return cvae

def get_grid (X_lat, n_samples):
    x = np.linspace(np.min(X_lat[:, 0]), np.max(X_lat[:, 0]), n_samples)
    y = np.linspace(np.min(X_lat[:, 1]), np.max(X_lat[:, 1]), n_samples)
    xv, yv = np.meshgrid(x, y)

    grid_latents = np.vstack((np.reshape(xv, -1), np.reshape(yv, -1))).T

    grid_latents = grid_latents[::-1]
    grid_latents = np.reshape(grid_latents, (n_samples, n_samples, 2))
    grid_latents = grid_latents[:, ::-1]
    grid_latents = np.reshape(grid_latents, (n_samples ** 2, 2))

    return grid_latents

def save_trajectories(indices, nrn,nln,astar,inter, model_name, n_steps,
        object_configuration):
    indices_filename = output_dir + "trajectories/indices_"
    indices_filename += model_name
    indices_filename += "_steps_" + str(n_steps)
    indices_filename += "_object_configuration_" + str(object_configuration)

    nrn_trajectories_filename = output_dir + "trajectories/nrn_trajectories_"
    nrn_trajectories_filename += model_name
    nrn_trajectories_filename += "_object_configuration_" + str(object_configuration)
    
    nln_trajectories_filename = output_dir + "trajectories/nln_trajectories_"
    nln_trajectories_filename += model_name
    nln_trajectories_filename += "_object_configuration_" + str(object_configuration)
    
    astar_trajectories_filename = output_dir + "trajectories/astar_trajectories_"
    astar_trajectories_filename += model_name
    astar_trajectories_filename += "_object_configuration_" + str(object_configuration)
    
    inter_trajectories_filename = output_dir + "trajectories/inter_trajectories_"
    inter_trajectories_filename += model_name
    inter_trajectories_filename += "_steps_" + str(n_steps)
    inter_trajectories_filename += "_object_configuration_" + str(object_configuration)
    


    np.save(indices_filename, indices.astype('float64'))
    np.save(nrn_trajectories_filename, nrn)
    np.save(nln_trajectories_filename, nln)
    np.save(astar_trajectories_filename, astar)
    np.save(inter_trajectories_filename, inter)
    
    

