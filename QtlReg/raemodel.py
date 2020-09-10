from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784*3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1)

  

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu= self.encode(x.view(-1, 784))
        return self.decode(mu), mu