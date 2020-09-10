from __future__ import print_function
import argparse
import h5py
import numpy as np
import os
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import torchvision.utils as vutils
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import math
from sklearn.datasets import make_blobs
from scipy.ndimage import gaussian_filter
from VAE_model_pixel import Encoder, Decoder, VAE_Generator
pret = 0
random.seed(8)
input_size=64


def show_and_save(file_name, img):
    f = "results/%s.png" % file_name
    save_image(img[2:3, :, :], f, range=[0, 1.5])

    #fig = plt.figure(dpi=300)
    #fig.suptitle(file_name, fontsize=14, fontweight='bold')
    #plt.imshow(npimg)
    #plt.imsave(f,npimg)


def save_model(epoch, encoder, decoder):
    torch.save(decoder.cpu().state_dict(), 'results/VAE_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(), 'results/VAE_encoder_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()


def load_model(epoch, encoder, decoder, loc):
    #  restore models
    decoder.load_state_dict(torch.load(loc+'/VAE_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc+'/VAE_encoder_%d.pth' % epoch))
    encoder.cuda()
  

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

###### define constant########
input_channels = 3
hidden_size = 1
max_epochs = 40
lr = 3e-4
beta =0
dim1=1
#######network################
epoch=20
LM='/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/Brats_results'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
#load_model(epoch,G.encoder, G.decoder,LM)
opt_enc = optim.Adam(G.parameters(), lr=lr)


#######losss#################






def prob_loss_function(recon_x, var_x, x, mu, logvar):
    # x = batch_sz x channel x dim1 x dim2
    

    x_temp = x.repeat(dim1, 1, 1, 1)
    msk = torch.tensor(x_temp > 1e-6).float()
    mskvar = torch.tensor(x_temp < 1e-6).float()


    msk2 = torch.tensor(x_temp > -1).float()
    NDim = torch.sum(msk2,(1,2,3))
    std = var_x.mul(0.5).exp_()
    const = (-torch.sum(var_x*msk+mskvar, (1, 2, 3))) / 2



    term1 = torch.sum((((recon_x - x_temp)*msk / std)**2), (1, 2, 3))
    #const2 = -(NDim / 2) * math.log((2 * math.pi))

    prob_term = const + (-(0.5) * term1) #+const2
    BBCE = torch.sum(prob_term / dim1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   
   

    return -BBCE + KLD





################################

#if pret == 1:
   #oad_model(499, G.encoder, G.decoder)

##############train#####################
train_loss = 0
valid_loss = 0
pay=0
valid_loss_list, train_loss_list = [], []
for epoch in range(max_epochs):
    train_loss = 0
    valid_loss = 0
    for data in train_loader:
        batch_size = data.size()[0]

        #print (data.size())
        datav = Variable(data).cuda()
        #datav[l2,:,row2:row2+5,:]=0

        mean, logvar, rec_enc, var_enc = G(datav)
        if beta == 0:
            prob_err = prob_loss_function(rec_enc, var_enc, datav, mean,
                                          logvar)
        else:
           addhoc=0
        err_enc = prob_err
        opt_enc.zero_grad()
        err_enc.backward()
        opt_enc.step()
        train_loss += prob_err.item()
    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    G.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = Variable(data).cuda()
            mean, logvar, valid_enc, valid_var_enc = G(data)
            if beta == 0:
                prob_err = prob_loss_function(valid_enc, valid_var_enc, data,
                                              mean, logvar)
            else:
                addhoc=0
            valid_loss += prob_err.item()
        
            if i == 0:
                std_var_enc=valid_var_enc.mul(0.5).exp_()
                n = min(data.size(0), 8)
                comparison = torch.cat([
                    data[:n, [2], ],
                    valid_enc.view(args.batch_size, 3, 64, 64)[:n, [2], ],
                    std_var_enc.view(args.batch_size, 3, 64, 64)[:n, [2], ]           
                ])
                save_image(comparison.cpu(),
                           'results/recon_mean_' + str(epoch) + '_' +
                           '.png',
                           nrow=n)
        valid_loss /= len(test_loader.dataset)
    if epoch == 0:
        best_val = valid_loss
    elif (valid_loss < best_val):
        save_model(epoch, G.encoder, G.decoder)
        pay=0
        best_val = valid_loss
    pay=pay+1
    if(pay==100):
        break


    print(valid_loss)
    
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
 

    

    
######################################

save_model(epoch, G.encoder, G.decoder)
plt.plot(train_loss_list, label="train loss")
plt.plot(valid_loss_list, label="validation loss")
plt.legend()
plt.show()
