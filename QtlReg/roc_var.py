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
from VAE_model_pixel64_shrink_floorvar import VAE_Generator as VAE

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


model_median.load_state_dict(torch.load('results/VAE_QR_brain100.pth'))


out_median = torch.zeros(in_data.shape)
out_std = torch.zeros(in_data.shape)

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
        out_median[i, ] = rec_med

        out_std[i, ] = (Q15)

        # division by 2 to compensate for multiplication ny 2 in the std dev autoencoder code

np.savez('results/rec_QR_brain.npz',
         out_med=out_median,
         out_std=out_std,
         in_data=in_data)

z_score = (in_data - out_median) / out_std
z_score[z_score!=z_score]=0

p_value = torch.tensor(scipy.stats.norm.sf(z_score)).float()

p_value_orig = p_value.clone()

for ns in tqdm(range(p_value.shape[0])):
    fdrres = multipletests(p_value[ns, 2, :, :].flatten(),
                           alpha=0.05,
                           method='fdr_bh',
                           is_sorted=False,
                           returnsorted=False)
    p_value[ns, 2, :, :] = torch.tensor(fdrres[1]).reshape((64, 64))

    p_value[ns, 2, :, :] = torch.tensor(fdrres[1]).reshape((64, 64))

msk = (in_data.clone().detach() > .01).float()
p_value = p_value * msk + (1 - msk)

#z_score = z_score * msk + (1 - msk)

y_probas = z_score[:,2,:,:] #1-p_value[:,2,:,:] 

#fpr, tpr, th= metrics.roc_curve(y_true, y_probas)
y_probas = np.reshape(y_probas, (-1, 1,64,64))
y_probas = medfilt(y_probas,(1,1,7,7))
y_probas[y_probas>10^6]=0
y_probas[y_probas<0]=0
y_probas[y_probas!=y_probas]=0




fpr, tpr, th= metrics.roc_curve(y_true.flatten()[maskX.flatten()>-1e10]>0, y_probas.flatten()[maskX.flatten()>-1e10])
L=fpr/tpr
best_th=th[fpr<0.05]
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()

n = np.array(range(0, 16 * 16, 16))

pv = p_value[n].clone().detach()

sig_msk = (pv < 0.05).clone().detach().float()
comparison = torch.cat([
    in_data[n, 2:3], out_median[n, 2:3],
    3 * out_std[n, 2:3], 3 * abs(in_data[n, 2:3] - out_median[n, 2:3]),
    z_score[n, 2:3] / 3.0, sig_msk[:, 2:3]
])

save_image(comparison,
           'results/recon_QR_pval_brain.png',
           nrow=16,
           scale_each=False,
           normalize=True,
           range=(0, 1))
#|
#      (out_median.clone().detach() > .1)).float()

#sig_msk = ((in_data > out_Q1) | (in_data < out_Q2)).clone().detach().float()


input("Press Enter to continue...")
