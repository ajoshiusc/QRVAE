from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from keras.datasets import mnist
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from vaemodel_brain import VAE_Generator as VAE
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=50,
                    metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='enables CUDA training')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=40,
    metavar='N',
    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


#(x_train, _), (x_test, _) = mnist.load_data()

(X, _), (_, _) = mnist.load_data()
X = X / 255
X = X.astype(float)
#x_train, x_test = train_test_split(X)

d = np.load(
    '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data__maryland_histeq.npz')
X = d['data']
X = X[0:2380, :, :, :]


d = np.load(
    '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data__TBI_histeq.npz')
X = np.concatenate((X, d['data']))

X = np.transpose(X[:, ::2, ::2, :], (0, 3, 1, 2))

'''x_train = x_train / 255
x_train = x_train.astype(float)
x_test = x_test / 255
x_test = x_test.astype(float)'''

in_data = X  # np.concatenate((x_train, x_test), axis=0)
in_data = torch.tensor(in_data).float()

#x_train = torch.from_numpy(x_train).float().view(x_train.shape[0],1,28,28)
#x_test = torch.from_numpy(x_test).float().view(x_test.shape[0],1,28,28)

input_channels = 3
hidden_size = 128

model = VAE(input_channels, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.load_state_dict(torch.load('results/VAE_mean_brain.pth'))

out_data = np.zeros(in_data.shape)

model.eval()
with torch.no_grad():

    for i, data in enumerate(tqdm(in_data)):
        data = data[None, ].to(device)
        mean, logvar, rec = model(data)
        out_data[i, ] = rec.cpu()

np.savez('results/rec_data_brain.npz', out_data=out_data, in_data=in_data)
