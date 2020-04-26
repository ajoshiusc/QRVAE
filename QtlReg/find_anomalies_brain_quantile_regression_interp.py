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
from vaemodel_brain import VAE_Generator as VAE
from utils import make_lesion
from statsmodels.stats.multitest import multipletests
#from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from itertools import product

torch.no_grad()

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

quantiles = np.arange(1e-6, 1 - 1e-6, .05)
model = VAE(input_channels, hidden_size).to(device)
out_Q = np.zeros((len(quantiles), *list(in_data.shape)))

for qt, Q in enumerate(tqdm(quantiles)):  #[0.15, 0.5, 0.85]:

    model.load_state_dict(
        torch.load('results/VAE_QR_brain' + '_' + str(Q) + '.pth'))

    model.eval()

    for i, data in enumerate(in_data):
        """     add artificial lesion
                data[0, :, :] = data[0, :, :] + \
            torch.tensor(make_lesion(data[0, :, :]))
        """
        data = data[None, ].to(device)
        mean, logvar, rec_Q = model(data)
        out_Q[qt, i, ] = rec_Q.cpu().detach()

        # division by 2 to compensate for multiplication ny 2 in the std dev autoencoder code

#out_Q = out_Q.detach().numpy()
in_data = in_data.numpy()

np.savez('results/rec_QR_all_brain.npz', out_Q=out_Q, in_data=in_data)

#f = interp1d(quantiles, out_Q, axis=0)
#p_value = f(in_data)

#loops on all dimensions
Qr = range(out_Q.shape[1])
Mr = range(out_Q.shape[2])
Xr = range(out_Q.shape[3])
Yr = range(out_Q.shape[4])

p_value = np.zeros(in_data.shape)

for d in tqdm(Qr):
    for m, x, y in product(Mr, Xr, Yr):
        p_value[d, m, x, y] = griddata(out_Q[:, d, m, x, y],
                                       quantiles,
                                       in_data[d, m, x, y],
                                       method='linear',
                                       fill_value=0.5)

#p_value = torch.tensor(scipy.stats.norm.sf(z_score)).float()
'''
p_value_orig = p_value.copy()

for ns in tqdm(range(p_value.shape[0])):
    fdrres = multipletests(p_value[ns, 2, :, :].flatten(),
                           alpha=0.05,
                           method='fdr_bh',
                           is_sorted=False,
                           returnsorted=False)
    p_value[ns, 2, :, :] = torch.tensor(fdrres[1]).reshape((64, 64))

    p_value[ns, 2, :, :] = torch.tensor(fdrres[1]).reshape((64, 64))

'''
msk = ((in_data > .01))
p_value = p_value * msk + (1 - msk)

n = np.array(range(0, 16 * 16, 16))

pv = p_value[n]

pv[np.isnan(pv)] = 1
sig_msk = (pv < 0.05).astype(np.float32)
comparison = torch.cat([
    torch.tensor(in_data[n, 2:3]),
    torch.tensor(out_Q[10, n, 2:3].astype(np.float32)),
    torch.tensor(out_Q[1, n, 2:3].astype(np.float32)),
    torch.tensor(out_Q[19, n, 2:3].astype(np.float32)),
    torch.tensor(sig_msk[:, [2]])
])

save_image(comparison,
           'results/recon_sig_all_brain.png',
           nrow=16,
           scale_each=False,
           normalize=True,
           range=(0, 1))

input("Press Enter to continue...")
