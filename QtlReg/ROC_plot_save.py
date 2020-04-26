from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import json
import numpy as np
import math
import datetime
import os
from keras.datasets import mnist
from torchsummary import summary
import matplotlib.pyplot as plt
#import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn import metrics


import VAE_models
from sklearn.model_selection import train_test_split

seed = 10009
epochs = 250
batch_size = 8
log_interval = 10
beta=0
sigma=0.2
z=32

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=10,
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
                    help='random seed (import matplotlib.pyplot as pltult: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='how many batches to wait before logging training status')
    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#print(y_test[1:10])
d=np.load('/big_disk/akrami/git_repos/lesion-detector/src/VAE/data_24_ISEL.npz')
X=d['data']
X_data=X[:,:,:,0:3]
max_val=np.max(X_data,1)
max_val=np.max(max_val,1)
max_val=np.reshape(max_val,(-1,1,1,3))
X_data= X_data/ max_val
X_data = X_data.astype('float64')
D=X_data.shape[1]*X_data.shape[2]



X_data = np.transpose(X_data , (0, 3, 1,2))

PATH='/big_disk/akrami/git_repos/lesion-detector/src/VAE/models/model_%d_%d.pt' % (z,beta)
test_loader = torch.utils.data.DataLoader(X_data,
                                          batch_size=batch_size,
                                          shuffle=False)


#####test data####
#x_test=np.load('x_test.npy')
#x_test = x_test/ 255
#x_test = x_test.astype('float64')
#y_test=y_test.astype('float64')
#x_test=x_test.reshape((x_test.shape[0],1,x_test.shape[1],x_test.shape[2] ))

#test_data = torch.from_numpy(x_test).float()
#test_data = test_data.to('cuda') if args.cuda else test_data.to('cpu')

#test_loader = torch.utils.data.DataLoader(test_data,
                                          #batch_size=batch_size,
                                         #shuffle=False)
model = VAE_models.VAE_nf(z)
model.load_state_dict(torch.load(PATH))
#model.have_cuda = args.cuda

if args.cuda:
    model.cuda()
model.eval()
#model.have_cuda = args.cuda



# Reconstruction + KL divergence losses summed over all elements and batch
def MSE_loss(Y, X):
    ret = (X- Y) ** 2
    ret = torch.sum(ret,1)
    return ret 
def BMSE_loss(Y, X, beta,sigma,D):
    term1 = -((1+beta) / beta)
    K1=1/pow((2*math.pi*( sigma** 2)),(beta*D/2))
    term2=MSE_loss(Y, X)
    term3=torch.exp(-(beta/(2*( sigma** 2)))*term2)
    loss1=torch.sum(term1*(K1*term3-1))
    return loss1


# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128*128*3), x.view(-1, 128*128*3), beta,sigma,D)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 128*128*3),x.view(-1, 128*128*3)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE + KLD

def test(X):
    test_loss = 0
    ind=0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            #        data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
            data.to(device=device, dtype=torch.float)
            temp=X[ind:ind+batch_size,:,:,3]
            seg=torch.from_numpy(temp.astype('float64'))
            ind=ind+batch_size
            seg = (seg).to(device)
            recon_batch, mu, logvar = model(data)
            #print(mu.shape)
            test_loss += beta_loss_function(recon_batch, data, mu,
                                            logvar,beta).item()
            if i == 0:
                f_data=data[:,2,:,:]
                f_recon_batch=recon_batch[:,2,:,:]
                rec_error=f_data-f_recon_batch
                n = min(f_data.size(0), 100)
                comparison = torch.cat([
                    f_data.view(batch_size, 1, 128, 128)[:n],
                    f_recon_batch.view(batch_size, 1, 128, 128)[:n],
                    (f_data.view(batch_size, 1, 128, 128)[:n]-f_recon_batch.view(batch_size, 1, 128, 128)[:n]),
                    torch.abs(f_data.view(batch_size, 1, 128, 128)[:n]-f_recon_batch.view(batch_size, 1, 128, 128)[:n]),
                    seg.view(batch_size, 1, 128, 128)[:n]

                ])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png',
                           nrow=n)
                mu_all=mu
                logvar_all=logvar
                rec_error_all=rec_error
            else:
                rec_error_all=torch.cat([
                    rec_error_all,
                    rec_error
                ])
                mu_all=torch.cat([
                    mu_all,
                    mu
                ])
                logvar_all=torch.cat([
                    logvar_all,
                    logvar
                ])
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return logvar_all,mu_all,test_loss,rec_error_all
if __name__ == "__main__":
    logvar_all,mu_all,test_loss,rec_error_all=test(X)
    y_true = X[:,:,:,3]
    y_true=y_true.view(-1,1)
    y_probas = rec_error_all.veiw(-1,1)
    y_probas=y_probas.numpy()
        #metrics.plot_roc_curve(y_true, y_probas)
    plt.show()
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas, pos_label=2)
    metrics.auc(fpr, tpr)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()