from __future__ import print_function

#import multiprocessing
#multiprocessing.set_start_method('spawn', True)
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from keras.datasets import mnist
from vaemodel import VAE
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='VAE MNIST Example')
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
    default=50,
    metavar='N',
    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

(X, _), (_, _) = mnist.load_data()
X = X / 255
X = X.astype(float)
x_train, x_test = train_test_split(X, test_size=0.25)
dim=784

#x_test = x_test / 255
#x_test = x_test.astype(float)
x_train = torch.from_numpy(x_train).float().view(x_train.shape[0], 1, 28, 28)
x_test = torch.from_numpy(x_test).float().view(x_test.shape[0], 1, 28, 28)

train_loader = torch.utils.data.DataLoader(x_train,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           **kwargs)
test_loader = torch.utils.data.DataLoader(x_test,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          **kwargs)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    dim=784
    x=x.view(-1, dim)
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    log_px_Q1 = torch.sum(torch.max(0.15 * (recon_x[:,0:dim] - x), (0.15 - 1) * (recon_x[:,0:dim] - x)).view(-1, dim),(1))
    log_px_Q2 = torch.sum(torch.max(0.5 * (recon_x[:,dim:dim*2] - x), (0.5 - 1) * (recon_x[:,dim:dim*2] - x)).view(-1, dim),(1))
    log_px_Q3= torch.sum(torch.max(0.85 * (recon_x[:,dim*2:dim*3] - x), (0.85 - 1) * (recon_x[:,dim*2:dim*3] - x)).view(-1, dim),(1))
    log_px=(log_px_Q1+log_px_Q2+log_px_Q3)/3

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return log_px.sum() + 0.5* KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
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


def test(epoch):
    model.eval()
   
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([
                    data[:n],
                    recon_batch[:,dim:dim*2].view(args.batch_size, 1, 28, 28)[:n],
                    (torch.abs(recon_batch[:,dim*2:dim*3]-recon_batch[:,dim:dim*2])/2).view(args.batch_size, 1, 28, 28)[:n],

                ])
                save_image(comparison.cpu(),
                           'results/recon_mean_' + str(epoch) + '.png',
                           nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample[:,dim:dim*2].view(64, 1, 28, 28),
                       'results/sample_med_' + str(epoch) + '.png')

    print('saving the model')
    torch.save(model.state_dict(), 'results/VAE_QR.pth')
    print('done')

    input("Press Enter to continue...")
