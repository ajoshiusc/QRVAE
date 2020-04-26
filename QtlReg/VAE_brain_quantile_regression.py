from __future__ import print_function

# import multiprocessing
# multiprocessing.set_start_method('spawn', True)
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from keras.datasets import mnist
from vaemodel_brain import VAE_Generator as VAE
from sklearn.model_selection import train_test_split
import numpy as np

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=400, # 250
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
    default=50,
    metavar='N',
    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

d = np.load(
    '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data__maryland_histeq.npz'
)
X = d['data']
X = X[0:2380, :, :, :]
X_train = X[0:-20 * 20, :, :, :]
X_valid = X[-20 * 20:, :, :, :]

d = np.load(
    '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data__TBI_histeq.npz'
)
X_train = np.concatenate((X_train, d['data'][0:-20 * 20, :, :, :]))
X_valid = np.concatenate((X_valid, d['data'][-20 * 20:, :, :, :]))

x_train = np.transpose(X_train[:, ::2, ::2, :], (0, 3, 1, 2))
x_test = np.transpose(X_valid[:, ::2, ::2, :], (0, 3, 1, 2))
"""
(X, _), (_, _) = mnist.load_data()
X = X / 255
X = X.astype(float)
x_train, x_test = train_test_split(X, test_size=0.25)

# x_test = x_test / 255
# x_test = x_test.astype(float)
x_train = torch.from_numpy(x_train).float().view(x_train.shape[0], 1, 28, 28)
x_test = torch.from_numpy(x_test).float().view(x_test.shape[0], 1, 28, 28)

"""

train_loader = torch.utils.data.DataLoader(x_train.astype(np.float32),
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           **kwargs)
test_loader = torch.utils.data.DataLoader(x_test.astype(np.float32),
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          **kwargs)
input_channels = 3
hidden_size = 128

model = VAE(input_channels, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, Q=0.5):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    #MSE = F.mse_loss(recon_x, x, reduction='sum')
    MSE = torch.sum(torch.max(Q * (recon_x - x), (Q - 1) * (recon_x - x)))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


def train(epoch, Q=0.5):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        mu, logvar, recon_batch = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, Q)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch, Q=0.5):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            mu, logvar, recon_batch = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([
                    data[:n, [2], ],
                    recon_batch.view(args.batch_size, 3, 64, 64)[:n, [2], ]
                ])
                save_image(comparison.cpu(),
                           'results/recon_mean_' + str(epoch) + '_' + str(Q) +
                           '.png',
                           nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":

    for Q in [0.15, 0.5, 0.85]: #np.arange(1e-6,1-1e-6,.05):#
        for epoch in range(args.epochs):
            train(epoch, Q)
            '''test(epoch, Q)
            with torch.no_grad():
                sample = torch.randn(64, hidden_size).to(device)
                sample = model.decoder(sample)
                save_image(
                    sample[:, 2, :, :].view(64, 1, 64,
                                            64), 'results/sample_mean_' +
                    str(epoch) + '_' + str(Q) + '.png')'''

        print('saving the model')
        torch.save(model.state_dict(),
                   'results/VAE_QR_brain' + '_' + str(Q) + '.pth')
        print('done')

    input("Press Enter to continue...")
