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
                    default=200, # 250
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
hidden_size = 128
max_epochs =100
lr = 3e-4
beta =0

##########load low res net##########
model=UNet(3, 6).cuda()
#load_model(epoch,G.encoder, G.decoder,LM)
optimizer = optim.Adam(model.parameters(), lr=lr)

#######losss#################
def prob_loss_function(recon_x, var_x, x, mu, logvar):
    # x = batch_sz x channel x dim1 x dim2
    
   x_temp = x.repeat(10, 1, 1, 1)
    msk = torch.tensor(x_temp > 1e-6).float()
    NDim = torch.sum(msk)

    std = logvar_x.mul(0.5).exp_()
    #std_all=torch.prod(std,dim=1)
    const = torch.sum(logvar_x * msk, (1, 2, 3)) / 2
    #const=const.repeat(10,1,1,1) ##check if it is correct
    const2 = (NDim / 2) * math.log((2 * math.pi))

    term1 = (0.5) * torch.sum((((recon_x - x_temp) * msk / std)**2), (1, 2, 3))

    #term2=torch.log(const+0.0000000000001)
    term2 = -(beta / (beta + 1)) * torch.sum(logvar_x.mul(0.5)*msk, (1, 2, 3))
    term3 = -(1 / (beta + 1)) * 0.5 * NDim* (beta * math.log(
        ((2 * math.pi))) + math.log(beta + 1))
    prob_term = const + const2 + (term1)  + term2 + term3

    BBCE = torch.sum(prob_term / 10)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   

    return -BBCE+0.05*KLD 

# Reconstruction + KL divergence losses summed over all elements and batch



def train(epoch, Q=0.5):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        reg_weight=0.001
        mu, logvar, recon_batch = model(data)

        regularize_loss=model.sparse_loss()
        mean_x=recon_batch[:,0:3,:,:]
        var_x=(recon_batch[:,3:6,:,:])
        loss = prob_loss_function(mean_x, var_x, data, mu, logvar)+regularize_loss*reg_weight
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
            mean_x=recon_batch[:,0:3,:,:]
            var_x=recon_batch[:,3:6,:,:] 
            test_loss += prob_loss_function(mean_x, var_x, data, mu, logvar)
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([
                    data[:n, [2], ],
                    recon_batch.view(args.batch_size, 6, 64, 64)[:n, [2], ],
                    ((recon_batch.view(args.batch_size, 6, 64, 64)[:n, [5], ]).mul(0.5).exp_())*3        
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
