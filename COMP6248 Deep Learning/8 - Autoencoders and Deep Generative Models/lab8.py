import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from itertools import chain

import numpy as np

import matplotlib.pyplot as plt 

device = "cpu"
batch_size = 256
image_dim = 784 #flattened

# dataset construction
transform = transforms.Compose([
    transforms.ToTensor(), # convert to tensor
    transforms.Lambda(lambda x: x.view(image_dim)) # flatten into vector
    ])

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    ,train=True
    ,download=True
    ,transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size
)


class VAE_Encoder(nn.Module):
    '''
    simple encoder with a single hidden dense layer (ReLU activation)
    and linear projections to the diag-Gauss parameters
    '''
    def __init__(self, image_dim, enc_hidden_units, embedding_dim):
        super(VAE_Encoder, self).__init__()
        self.fc1 = nn.Linear(image_dim, enc_hidden_units)
        self.mu = nn.Linear(enc_hidden_units, embedding_dim)
        self.log_sigma = nn.Linear(enc_hidden_units, embedding_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        mu = self.mu(out)
        log_sigma = self.log_sigma(out)
        return (mu, log_sigma)

class VAE_Decoder(nn.Module):
    '''
    simple decoder: single dense hidden layer (ReLU activation) followed by 
    output layer with a sigmoid to squish values
    '''
    def __init__(self, embedding_dim, dec_hidden_units, image_dim):
        super(VAE_Decoder, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, dec_hidden_units)
        self.fc2 = nn.Linear(dec_hidden_units, image_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out
		
# Sampling function (using the reparameterisation trick)
def sample(mu, log_sigma2):
    eps = torch.randn(mu.shape[0], mu.shape[1]).to(device)
    return mu + torch.exp(log_sigma2 / 2) * eps


#parameters
batch_size = 256
embedding_dim = 2
enc_hidden_units = 512
dec_hidden_units = 512
nEpoch = 30

# construct the encoder, decoder and optimiser
enc = VAE_Encoder(image_dim, enc_hidden_units, embedding_dim)
dec = VAE_Decoder(embedding_dim, dec_hidden_units, image_dim).to(device)
optimizer = optim.Adam(chain(enc.parameters(), dec.parameters()), lr=1e-3)

# training loop
for epoch in range(nEpoch):
    losses = []
    trainloader = tqdm(train_loader)

    for i, data in enumerate(trainloader, 0):
        inputs, _ = data

        optimizer.zero_grad()

        mu, log_sigma2 = enc(inputs)
        z = sample(mu, log_sigma2)
        outputs = dec(z)

        # E[log P(X|z)] - as images are binary it makes most sense to use binary cross entropy
        # we need to be a little careful - by default torch averages over every observation 
        # (e.g. each  pixel in each image of each batch), whereas we want the average over entire
        # images instead
        recon = F.binary_cross_entropy(outputs, inputs, reduction='sum') / inputs.shape[0]
        
        kl = 0 
        # kl = D_KL(Q(z|X) || P(z|X)) - calculate in closed form
        # Compute the term kl which is then added to the total loss
        kl = torch.sum(log_sigma2.exp() + mu.pow(2) - 1 - log_sigma2) / 2
        kl /= batch_size * image_dim
        
        loss = recon + kl
        loss.backward()
        optimizer.step()

        # keep track of the loss and update the stats
        losses.append(loss.item())
        trainloader.set_postfix(loss=np.mean(losses), epoch=epoch)

torch.save(dec.state_dict(), "VAE_weights30epoch")
#dec.load_state_dict(torch.load("VAE_weights30epoch"))

def plot_reconstructed(decoder, r0=(-4, 4), r1=(-4, 4), n=21, filename="VAE.png"):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            print(z)
            x_hat = decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1], cmap=plt.get_cmap('gray'))
    plt.savefig(filename)

plot_reconstructed(dec)

class Encoder(nn.Module):
    '''
    simple encoder with no hidden dense layer
    '''
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.dense = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        out = self.dense(x)
        out = F.relu(out)
        return out

class Decoder(nn.Module):
    '''
    simple decoder: single dense hidden layer followed by 
    output layer with a sigmoid to squish values
    '''
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.dense(x)
        out = F.sigmoid(out)
        return out
		

enc_dim = 2
image_dim = 784  # [flattened]
nEpoch = 0

# construct the encoder, decoder and optimiser
enc = Encoder(image_dim, enc_dim)
dec = Decoder(enc_dim, image_dim).to(device)
optimizer = optim.Adam(chain(enc.parameters(), dec.parameters()), lr=1e-3)

# training loop
for epoch in range(nEpoch):
    losses = []
    trainloader = tqdm(train_loader)

    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        optimizer.zero_grad()

        z = enc(inputs)
        outputs = dec(z)

        loss = F.binary_cross_entropy(outputs, inputs, reduction='sum') / inputs.shape[0]
        loss.backward()
        optimizer.step()

        # keep track of the loss and update the stats
        losses.append(loss.item())
        trainloader.set_postfix(loss=np.mean(losses), epoch=epoch)
    
#torch.save(dec.state_dict(), "AE_weights30epoch")
dec.load_state_dict(torch.load("AE_weights30epoch"))

plot_reconstructed(dec, filename="AE.png")