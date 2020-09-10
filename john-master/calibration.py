#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 08:11:10 2019

@author: nsde
"""
#%%
import torch
from torch import nn
from torch import distributions as D
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from utils import batchify, dist, translatedSigmoid, RBF, PosLinear, Reciprocal
from itertools import chain
from locality_sampler import gen_Qw, locality_sampler, get_pseupoch, local_batchify
from sklearn.cluster import KMeans
sns.set()

#%%
n_neurons = 50

#%%
class basemodel(nn.Module):
    def __init__(self, latent_size=2, cuda=True):
        super(basemodel, self).__init__()
        self.switch = 0.0
        self.latent_size = latent_size
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        
    def encoder(self, x):
        return self.enc_mu(x), self.enc_var(x)

    def decoder(self, z):
        x_mu = self.dec_mu(z)
        #x_var = self.switch * x_var + (1-self.switch)*torch.tensor([0.02**2], device=z.device)
        return x_mu

    def sample(self, N):
        z = torch.randn(N, self.latent_size, device=self.device)
        x_mu= self.decoder(z)
        return x_mu
    
    def forward(self, x, beta=1.0, epsilon=1e-5,Q=0.5):
        z_mu, z_var = self.encoder(x)
        q_dist = D.Independent(D.Normal(z_mu, z_var.sqrt()+epsilon), 1)
        z = q_dist.rsample()
        x_mu = self.decoder(z) 
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        log_px_Q1 = torch.sum(torch.max(0.15 * (x-x_mu[:,0:4]), (0.15 - 1) * (x-x_mu[:,0:4])).view(-1, 4),(1))
        log_px_Q2 = torch.sum(torch.max(0.5 * (x-x_mu[:,4:8]), (0.5 - 1) * (x-x_mu[:,4:8])).view(-1, 4),(1))
        log_px_Q3= torch.sum(torch.max(0.85 * (x-x_mu[:,8:12] ), (0.85 - 1) * (x-x_mu[:,8:12] )).view(-1, 4),(1))
        log_px=(log_px_Q1+log_px_Q2+log_px_Q3)/3
        kl = q_dist.log_prob(z) - prior.log_prob(z)
        elbo = -log_px - 0.28*kl
        return elbo.mean(), log_px.mean(), kl.mean(), x_mu, z, z_mu, z_var
    
    def evaluate(self, X, L=1000):
        with torch.no_grad():
            x_mu, x_var = self.sample(L)
            parzen_dist = D.Independent(D.Normal(x_mu, x_var.sqrt()), 1)
            elbolist, logpxlist, parzen_score = [ ], [ ], [ ]
            for x in tqdm(X, desc='evaluating', unit='samples'):
                x = torch.tensor(x.reshape(1, -1), device=self.device)
                elbo, logpx, kl, _, _, _, _, _ = self.forward(x)
                elbolist.append(elbo.item())
                logpxlist.append(logpx.item())
                parzen_score.append(parzen_dist.log_prob(x).mean().item())
            
            return np.array(elbolist), np.array(logpxlist), np.array(parzen_score)
        
#%%
class vae(basemodel):
    def __init__(self, latent_size=2, cuda=True):
        super(vae, self).__init__(latent_size, cuda)
        
        self.enc_mu = nn.Sequential(nn.Linear(4, n_neurons),
                                    nn.ReLU(),
                                    nn.Linear(n_neurons, self.latent_size))
        self.enc_var = nn.Sequential(nn.Linear(4, n_neurons),
                                     nn.ReLU(),
                                     nn.Linear(n_neurons, self.latent_size),
                                     nn.Softplus())
        self.dec_mu = nn.Sequential(nn.Linear(self.latent_size, n_neurons),
                                    nn.ReLU(),
                                    nn.Linear(n_neurons, 4*3))
        
    
    def fit(self, Xtrain, n_iters=100, lr=1e-3, batch_size=256, beta=1.0):
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        it = 0
        batches = batchify(Xtrain, batch_size = batch_size, shuffel=True)
        progressBar = tqdm(desc='training', total=n_iters, unit='iter')
        loss = [ ]
        while it < n_iters:
            optimizer.zero_grad()
            x = torch.tensor(next(batches)[0], device=self.device)
            elbo, log_px, kl, x_mu, z, z_mu, z_var = self.forward(x)
            (-elbo).backward()
            optimizer.step()
            progressBar.update()
            progressBar.set_postfix({'elbo': (-elbo).item()})
            loss.append((-elbo).item())
            it+=1
        progressBar.close()
        return loss

#%%

#%%
if __name__ == '__main__':
    modelname = 'vae'
    
    Xtrain = np.zeros((500, 4))
    with open('data/data_v2.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for i, row in enumerate(reader):
            if i!=0:
                Xtrain[i-1] = [float(r) for r in row[1:]]
    Xval = np.zeros((500, 4))
    with open('data/data_v2_eval.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for i, row in enumerate(reader):
            if i!=0:
                Xval[i-1] = [float(r) for r in row[1:]]
    Xtrain = Xtrain.astype('float32')
    Xval = Xval.astype('float32')
    

    use_cuda = False
    if modelname == 'vae':
        model = vae(cuda=use_cuda)
    if use_cuda:
        model.cuda();
    
    loss = model.fit(Xtrain, n_iters=10000, lr=1e-3, beta=1.0,)
    model.eval()
    _, _, _, x_mu,_, _, _ = model.forward(torch.tensor(Xtrain).to(model.device))
    _, _, _, x_mu_val,_, _, _ = model.forward(torch.tensor(Xval).to(model.device))

    std_val=(x_mu_val[:,8:12]-x_mu_val[:,0:4])/2
    data_nom_val=torch.tensor(Xval).to(model.device)
    E=torch.max(x_mu_val[:,0:4]-data_nom_val,data_nom_val-x_mu_val[:,8:12]).detach()
    E=E.numpy()
    Q1=np.quantile(E, 0.70, axis=0)
    #Q2=np.quantile(E, 0.15, axis=0)


      

    

    #%%
    def savefig(c):
        plt.savefig('figs/2m4d_' + modelname + '_' + str(c) + '.pdf', format='pdf')
        plt.savefig('figs/2m4d_' + modelname + '_' + str(c) + '.png', format='png')
    
    def set_axes(fig):
        for i in range(4):
            for j in range(4):
                if j>i:
                    fig.axes[i,j].axis([
                        Xtrain[:,j].min()-0.5, Xtrain[:,j].max()+0.5,
                        Xtrain[:,i].min()-0.5, Xtrain[:,i].max()+0.5
                        ])
                if i>j:
                    fig.axes[i,j].axis([
                        Xtrain[:,j].min()-0.5, Xtrain[:,j].max()+0.5,
                        Xtrain[:,i].min()-0.5, Xtrain[:,i].max()+0.5
                        ])
    
    # plot
    plt.close('all')
  
    fig1 = sns.PairGrid(pd.DataFrame(Xtrain))
    fig1 = fig1.map_upper(plt.scatter, edgecolor="w")
    fig1 = fig1.map_lower(sns.kdeplot, cmap="Blues_d")
    fig1 = fig1.map_diag(sns.kdeplot, lw=3, legend=False)
    set_axes(fig1)
    savefig(1)
      
    fig2 = sns.PairGrid(pd.DataFrame(Xval))
    fig2 = fig2.map_upper(plt.scatter, edgecolor="w")
    fig2 = fig2.map_lower(sns.kdeplot, cmap="Blues_d")
    fig2 = fig2.map_diag(sns.kdeplot, lw=3, legend=False)
    set_axes(fig2)
    savefig(2)
      
   
    
    x_mu = model.sample(500)
    std=(x_mu[:,8:12]-x_mu[:,0:4])/2

    x_samp = x_mu[:,4:8] + (std**2) * torch.randn_like(x_mu[:,4:8])
    x_samp = x_samp[std.sum(dim=1) < 10]
    fig7 = sns.PairGrid(pd.DataFrame(x_samp.detach().numpy()))
    fig7 = fig7.map_upper(plt.scatter, edgecolor="w")
    fig7 = fig7.map_lower(sns.kdeplot, cmap="Blues_d")
    fig7 = fig7.map_diag(sns.kdeplot, lw=3, legend=False)
    set_axes(fig7)
    savefig(7)
    
    std=(x_mu[:,8:12]-x_mu[:,0:4])/2+(torch.tensor(Q1).to(model.device))
    x_samp = x_mu[:,4:8] + (std**2) * torch.randn_like(x_mu[:,4:8])
    x_samp = x_samp[std.sum(dim=1) < 10]
    fig8 = sns.PairGrid(pd.DataFrame(x_samp.detach().numpy()))
    fig8 = fig8.map_upper(plt.scatter, edgecolor="w")
    fig8 = fig8.map_lower(sns.kdeplot, cmap="Blues_d")
    fig8= fig8.map_diag(sns.kdeplot, lw=3, legend=False)
    set_axes(fig8)
    savefig(8)
    
    