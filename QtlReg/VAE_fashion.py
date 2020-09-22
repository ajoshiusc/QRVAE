from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
import scipy.io as spio
import matplotlib.pyplot as plt
import math



######initialize parameters#####
################################
seed = 10004
epochs = 150
batch_size = 120
log_interval = 10
CODE_SIZE = 20
np.random.seed(seed)
device = torch.device("cuda" )
kwargs = {'num_workers': 1, 'pin_memory': True} if device=="cuda" else {}
################################


#######read MNIST data set #############
##read MNIST data set and normalize it## 
##between zero and 1                  ##
##devide to train and validation      ##
########################################
def create_data():

    torch.manual_seed(seed)
    np.random.seed(seed)

    (X, X_lab), (_test_images, _test_lab) = fashion_mnist.load_data()
    X_lab = np.array(X_lab)

    # find bags
    ind = np.isin(X_lab, (0, 1, 2, 3, 4, 5, 6, 8))  #(1, 5, 7, 9)
    X_lab_outliers = X_lab[ind]
    X_outliers = X[ind]

    # find sneaker and ankle boots
    ind = np.isin(X_lab, (0,1,3,2,7, 9))  # (0, 2, 3, 4, 6))  #
    X_lab = X_lab[ind]
    X = X[ind]
    X = X / 255.0


    X = np.clip(X, 0, 1)
    X_train, X_valid, X_lab_train, X_lab_valid = train_test_split(
        X, X_lab, test_size=0.33, random_state=10003)
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_valid = X_valid.reshape((len(X_valid), np.prod(X_valid.shape[1:])))

    train_data = []
    for i in range(len(X_train)):
        train_data.append(
            [torch.from_numpy(X_train[i]).float(), X_lab_train[i]])

    test_data = []
    for i in range(len(X_valid)):
        test_data.append(
            [torch.from_numpy(X_valid[i]).float(), X_lab_valid[i]])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=len(test_data),
                                              shuffle=False)

    return train_loader, test_loader
###############################



class RVAE(nn.Module):
    def __init__(self):
        super(RVAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, CODE_SIZE)
        self.fc22 = nn.Linear(400, CODE_SIZE)
        self.fc3 = nn.Linear(CODE_SIZE, 400)
        self.fc4 = nn.Linear(400, 784)
        self.fc5 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        mu_out=torch.sigmoid(self.fc4(h3))
        sigma_out=torch.sigmoid(self.fc5(h3))
        return mu_out,sigma_out

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        mu_out,sigma_out=self.decode(z)
        return  mu, logvar,mu_out ,sigma_out

    def weight_reset(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()




model = RVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
def prob_loss_function(recon_x, var_x, x, mu, logvar):
    

    # x = batch_sz x channel x dim1 x dim2
    dim1=1
    x_temp = x.repeat(dim1, 1, 1)
    msk = torch.tensor(x_temp > 1e-6).float()
    mskvar = torch.tensor(x_temp < 1e-6).float()


    msk2 = torch.tensor(x_temp > -1).float()
    NDim = torch.sum(msk2,(1,2))
    std = var_x**(0.5)
    const = (-torch.sum(var_x.log()*msk+mskvar, (1, 2))) / 2



    term1 = torch.sum((((recon_x - x_temp)*msk / std)**2), (1, 2))
    #const2 = -(NDim / 2) * math.log((2 * math.pi))

    prob_term = const + (-(0.5) * term1) #+const2
    BBCE = torch.sum(prob_term / dim1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   

    return -BBCE+KLD


def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, data_lab) in enumerate(train_loader):

        data = (data).to(device)

        optimizer.zero_grad()
        mean, logvar, rec_enc,var_enc = model(data)
        #var_enc=torch.from_numpy(np.zeros((1)))
        loss = prob_loss_function(rec_enc, var_enc, data, mean,
                                          logvar)
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

    return(train_loss/ len(train_loader.dataset))


def model_reset():
    model.weight_reset()


def test():
    model.eval()
    test_loss = 0
 
    with torch.no_grad():
        for i, (data, data_lab) in enumerate(test_loader):
            data = (data).to(device)
            mean, logvar, rec_enc,var_enc = model(data)
            test_loss = prob_loss_function(rec_enc, var_enc, data, mean,
                                          logvar)

            if i == 0:
                n = min(data.size(0), 100)
                samp = [96, 97, 99, 90, 14, 35, 53, 57]
                comparison = torch.cat([
                    data.view(len(rec_enc,), 1, 28, 28)[samp],
                    rec_enc.view(len(rec_enc), 1, 28, 28)[samp],
                    (var_enc**(0.5)).view(len(rec_enc), 1, 28, 28)[samp]
                    
                ])

                save_image(comparison.cpu(),
                           'results/letters_mnist_recon_' + '.png',
                           nrow=n)


  
    test_loss /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}'.format(test_loss))
    print(var_enc.mean())

    return test_loss


if __name__ == "__main__":


    erange = range(1, epochs + 1)


    train_loss_list = []
    valid_loss_list=[]
    train_loader, test_loader = create_data()
    model_reset()
    for epoch in erange:

        train_loss=train(epoch)

        print("epoch: %d" % epoch)
        test_loss= test()
        train_loss_list.append(train_loss)
        valid_loss_list.append(test_loss)

    plt.plot(train_loss_list, label="train loss")
    plt.plot(valid_loss_list, label="validation loss")
    plt.legend()
    plt.show()