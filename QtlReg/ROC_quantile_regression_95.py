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
import scipy.stats
from vaemodel_brain_2out import VAE_Generator as VAE
from utils import make_lesion
from statsmodels.stats.multitest import multipletests
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.signal import medfilt

parser = argparse.ArgumentParser(description='VAE Brain Example')
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
    default=10,
    metavar='N',
    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
d = np.load(
    '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data_24_ISEL_histeq.npz'
)
X = d['data']
X_data = X[0:15 * 20, ::2, ::2, 0:3]

maskX = np.reshape(X[0:15*20, ::2, ::2, 2], (-1, 1))

y_true = X[0:15*20 ,::2, ::2, 3]

X_data = X_data.astype('float64')
X_valid = X_data[:, :, :, :]
D = X_data.shape[1] * X_data.shape[2]
####################################

##########train validation split##########
batch_size = 8
x_test = np.transpose(X_valid, (0, 3, 1, 2))

x_test = x_test.astype(float)

in_data = x_test
in_data = torch.tensor(in_data).float()

#x_train = torch.from_numpy(x_train).float().view(x_train.shape[0],1,28,28)
#x_test = torch.from_numpy(x_test).float().view(x_test.shape[0],1,28,28)

input_channels = 3
hidden_size = 128

model_median = VAE(input_channels, hidden_size).to(device)


model_median.load_state_dict(torch.load('results/VAE_QR_brain_0.5.pth'))



out_Q1 = torch.zeros(in_data.shape)
out_Q2 = torch.zeros(in_data.shape)

model_median.eval()



with torch.no_grad():

    for i, data in enumerate(tqdm(in_data)):
        """     add artificial lesion
                data[0, :, :] = data[0, :, :] + \
            torch.tensor(make_lesion(data[0, :, :]))
        """
        
        data = data[None, ].to(device)
        mean, logvar, rec_med,Q15 = model_median(data)
        msk = torch.tensor(data> 1e-6).float()        
        Q15=Q15*msk
        rec_med=rec_med*msk
        out_Q1[i, ]=Q15
        out_Q2[i, ] = rec_med



        # division by 2 to compensate for multiplication ny 2 in the std dev autoencoder code

msk = ((out_Q1.clone().detach() > .1)).float()




#|
#      (out_median.clone().detach() > .1)).float()
sig_msk = ((in_data <out_Q1) | (in_data >out_Q2)).clone().detach().float()
sig_msk = msk * sig_msk

n = np.array(range(0, 16 * 16, 16))
comparison = torch.cat([
    in_data[n, 2:3], out_Q1[n, 2:3], out_Q2[n, 2:3],
    sig_msk[n, 2:3]
])

save_image(comparison,
           'results/recon_QR_brain.png',
           nrow=16,
           scale_each=False,
           normalize=True,
           range=(0, 1))

input("Press Enter to continue...")
