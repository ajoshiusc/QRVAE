from __future__ import print_function
import argparse
import torch
import math
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import scipy.io as spio



#######initALIZATION#######
seed = 10004
epochs = 150  # was 20
batch_size = 120
log_interval = 10
CODE_SIZE = 20  # was 9
SIGMA = 1.0  # for Gaussian Loss function

##########################

########PARSERS###########

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=20,
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

device = torch.device("cuda" if args.cuda else "cpu")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#############################


#########DATA#############
dim=784
def create_data(frac_anom):

    torch.manual_seed(seed)
    np.random.seed(seed)

    (X, _), (_, _) = mnist.load_data()
    X = X / 255.0
    X = X.astype('float32')
    X_train, X_valid = train_test_split(X, test_size=0.25,random_state=10003)

    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_valid = X_valid.reshape((len(X_valid), np.prod(X_valid.shape[1:])))


    train_loader = torch.utils.data.DataLoader(X_train,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(X_valid,
                                              batch_size=len(X_valid),
                                              shuffle=False)

    return train_loader, test_loader
###############################


#############NET###############

class RVAE(nn.Module):
    def __init__(self):
        super(RVAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, CODE_SIZE)
        self.fc22 = nn.Linear(400, CODE_SIZE)
        self.fc3 = nn.Linear(CODE_SIZE, 400)
        self.fc4 = nn.Linear(400, 3*784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

   
    def weight_reset(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

    def sparse_loss(self):
        loss = 0
        for class_obj in self.modules():
            if isinstance(class_obj, nn.Linear) :
                    if class_obj.out_features >class_obj.in_features:
                        loss += torch.mean((class_obj.weight.data.clone()) ** 2)
        return loss

##############################


#############MODEL##############
model = RVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def model_reset():
    model.weight_reset()
##############################

############LOSS##############




def MSE_loss(Y, X):
    ret = (X - Y)**2
    ret = torch.sum(ret)
    return ret



# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x,mu,logvar,beta,Q):

    if beta > 0:
        # If beta is nonzero, use the beta entropy
        #BBCE = Gaussian_CE_loss(recon_x.view(-1, 784), x.view(-1, 784), beta)
        dummy=0
    else:
        # if beta is zero use binary cross entropy
        MSE = torch.sum(torch.max(Q * (recon_x.view(-1, 784) - x.view(-1, 784)), (Q - 1) * (recon_x.view(-1, 784) - x.view(-1, 784))))

    # compute KL divergence

   
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE+KLD
##########################################


#######################TRAIN##################
def train(epoch, beta_val):
    model.train()
    train_loss = 0
    #    for batch_idx, data in enumerate(train_loader):
    for batch_idx, (data) in enumerate(train_loader):
        #    for batch_idx, data in enumerate(train_loader):
        #data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
        data = (data).to(device)
        zeta_val=0
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        recon_batch=torch.reshape(recon_batch, (-1, 3,784))
        regularize_loss=model.sparse_loss()
        reg_weight=0.01

        loss_Q1 = beta_loss_function(recon_batch[:,0,:], data, mu,logvar, beta_val,0.85)
        loss_Q2= beta_loss_function(recon_batch[:,1,:], data, mu,logvar, beta_val,0.85)
        loss_Q3 = beta_loss_function(recon_batch[:,2,:], data, mu,logvar ,beta_val, 0.85)
        loss=(loss_Q1+loss_Q2+loss_Q3)/3+reg_weight*regularize_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
#####################TEST#########################



#############TEST################
def test(frac_anom, beta_val):
    model.eval()
    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
        
            data = (data).to(device)
            recon_batch, mu, logvar = model(data)
            recon_batch=torch.reshape(recon_batch, (-1, 3,784))
            
            if i == 0:
                n = min(data.size(0), 100)
                samp = np.arange(10)  # [
                #1 - 1, 101 - 1, 5 - 1, 7 - 1, 15 - 1, 109 - 1, 120 - 1,
                #   26 - 1, 30 - 1, 33 - 1
                # ]  #np.arange(200) #[4, 14, 50, 60, 25, 29, 32, 65]
                comparison = torch.cat([
                    data.view(len(recon_batch), 1, 28, 28)[samp],
                    recon_batch[:,0,:].view(len(recon_batch), 1, 28, 28)[samp],
                    recon_batch[:,1,:].view(len(recon_batch), 1, 28, 28)[samp],
                    recon_batch[:,2,:].view(len(recon_batch), 1, 28, 28)[samp],
                    (torch.abs(recon_batch[:,2,:]-recon_batch[:,0,:])/2).view(len(recon_batch), 1, 28, 28)[samp]

                ])
                save_image(comparison.cpu(),
                           'results/mnist_recon_shallow_' +
                           str(beta_val) + '_' + str(frac_anom) + '.png',
                           nrow=n)

                

    


    return 
#################################

if __name__ == "__main__":


    brange = np.array([0])
    erange = range(1, epochs + 1)
    anrange = np.array([0])
    
    for b, betaval in enumerate(brange):

        for a, frac_anom in enumerate(anrange):
            train_loader, test_loader = create_data(frac_anom)
            model_reset()
            for epoch in erange:

                train(epoch, beta_val=betaval)

                print('epoch: %d, beta=%g, frac_anom=%g' %
                      (epoch, betaval, frac_anom))

                test(frac_anom, beta_val=betaval)
            torch.save(model.state_dict(), 'results/VAE_QR.pth')

        



