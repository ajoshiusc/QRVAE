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
import random
from sklearn.model_selection import train_test_split
import numpy as np
from VAE import UNet
random.seed(4)


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size',
                    type=int,
                    default=8,
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
X_train = X[:, :, :, :]
#X_valid = X[-20 * 20:, :, :, :]

d = np.load(
    '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data__TBI_histeq.npz'
)
X_train = np.concatenate((X_train, d['data'][:, :, :, :]))

#X_train = np.concatenate((X_train, d['data'][0:-20 * 20, :, :, :]))
#X_valid = np.concatenate((X_valid, d['data'][-20 * 20:, :, :, :]))
d = np.load(
   '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data_24_ISEL_histeq.npz'
)
#X_train = np.concatenate((X_train, d['data'][15:24 * 20, :, :, 0:3]))
X_valid = d['data'][15:24 * 20, :, :, 0:3]


x_train = np.transpose(X_train[:, ::2, ::2, :], (0, 3, 1, 2))
x_test = np.transpose(X_valid[:, ::2, ::2, :], (0, 3, 1, 2))

torch.manual_seed(10)

train_loader = torch.utils.data.DataLoader(x_train.astype(np.float32),
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           **kwargs)
test_loader = torch.utils.data.DataLoader(x_test.astype(np.float32),
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          **kwargs)
input_channels = 3
hidden_size = 64
max_epochs =100
lr = 3e-4
beta =0

##########load low res net##########
model=UNet(3, 6).cuda()
#load_model(epoch,G.encoder, G.decoder,LM)
optimizer = optim.Adam(model.parameters(), lr=lr)

#######losss#################


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, Q=0.5):
    msk = torch.tensor(x> -1e-6).float()
    x=x*msk
    recon_x=recon_x*msk
    
    MSE = torch.mean(torch.sum(torch.max(Q * (x-recon_x ), (Q - 1) * (x-recon_x)).view(-1, 64*64*3),(1)))

    z_var=(1 + logvar - mu.pow(2) - logvar.exp()).view(-1, 8*8*128)
    KLD = -0.5 * torch.mean (torch.sum(z_var,(1)))

    return MSE+0.05* KLD


def train(epoch, Q=0.5):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        reg_weight=0.001
        mu, logvar, recon_batch = model(data)
        regularize_loss=model.sparse_loss()
        loss = (loss_function(recon_batch[:,0:3,:,:], data, mu,logvar, 0.15)+loss_function(recon_batch[:,3:6,:,:], data, mu,logvar, 0.5))/2+regularize_loss*reg_weight
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
            reg_weight=0.01
            test_loss += ((loss_function(recon_batch[:,0:3,:,:], data, mu,logvar, 0.15)+loss_function(recon_batch[:,3:6,:,:], data, mu,logvar, 0.5))/2).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([
                    data[:n, [2], ],
                    recon_batch.view(args.batch_size, 6, 64, 64)[:n, [2], ],
                    recon_batch.view(args.batch_size, 6, 64, 64)[:n, [5], ]           
                ])
                save_image(comparison.cpu(),
                           'results/recon_mean_' + str(epoch) + '_' + str(Q) +
                           '.png',
                           nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":

    #model.load_state_dict(torch.load('results/RAE_sum/RRAE_QR_brain_0.pth'))
    for epoch in range(args.epochs):
        train(epoch)
            
        test(epoch)
        if epoch % 5 ==0 :
            torch.save(model.state_dict(),
                   'results/VAE_QR_brain_'+str(epoch) +'.pth')
    print('done')

    input("Press Enter to continue...")
