import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class CVAE(nn.Module):
    def __init__(self, in_dim=21, hid_dim=64, lat_dim=2, c_dim=8):
        super(CVAE, self).__init__()

        self.input_shape = in_dim
        self.hidden_shape = hid_dim
        self.latent_shape = lat_dim

        ## Encoder
        self.fc1 = nn.Linear(self.input_shape + c_dim, self.hidden_shape)
        self.fc21 = nn.Linear(self.hidden_shape, self.latent_shape)
        self.fc22 = nn.Linear(self.hidden_shape, self.latent_shape)
        
        ## Decoder
        self.fc3 = nn.Linear(self.latent_shape + c_dim, self.hidden_shape)
        self.fc4 = nn.Linear(self.hidden_shape, self.input_shape)

    def encode(self, x, c):
        concat_input = torch.cat([x, c], 1)
        h1 = F.relu(self.fc1(concat_input))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c):
        z = torch.cat((z, c), dim=-1)
        h3 = F.relu(self.fc3(z))
        l = torch.nn.LeakyReLU(0.5)
        return l(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.input_shape), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
