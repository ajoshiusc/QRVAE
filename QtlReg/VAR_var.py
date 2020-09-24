# Copyright 2019 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function
import torch.utils.data
from scipy import misc
from torch import optim
from torchvision.utils import save_image
from net_var import *
import numpy as np
import pickle
import time
import random
import os
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from PIL import Image


im_size = 128


def loss_function(recon_x, var_x, x, mu, logvar):
    

    # x = batch_sz x channel x dim1 x dim2
    dim1=1
    x_temp = x.repeat(dim1, 1, 1, 1)
    msk = torch.tensor(var_x!=0).float()
    mskvar = torch.tensor(var_x==0).float()*1e-1

 
    std = (var_x+mskvar)**(0.5)
    const = (-torch.mean(var_x.log()*msk+mskvar, (1, 2, 3))) / 2



    term1 = torch.mean((((recon_x - x_temp) / std)**2), (1, 2, 3))
    #const2 = -(NDim / 2) * math.log((2 * math.pi))

    prob_term = const + (-(0.5) * term1) #+const2
    BBCE = torch.mean(prob_term / dim1)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
   

    return -BBCE+KLD

def process_batch(batch):
    data = [np.array(Image.fromarray(x).resize([im_size, im_size])).transpose((2, 0, 1)) for x in batch]

    x = torch.from_numpy(np.asarray(data, dtype=np.float32)).cuda() / 127.5 - 1.
    x = x.view(-1, 3, im_size, im_size)
    return x


def main():
    batch_size = 128
    z_size = 512
    vae = VAE(zsize=z_size, layer_count=5)
    vae.cuda()
    vae.train()
    vae.weight_init(mean=0, std=0.02)

    lr = 0.0005

    vae_optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
 
    train_epoch = 40

    sample1 = torch.randn(128, z_size).view(-1, z_size, 1, 1)

    for epoch in range(train_epoch):
        vae.train()

        with open('data_fold_%d.pkl' % (epoch % 5), 'rb') as pkl:
            data_train = pickle.load(pkl)

        print("Train set size:", len(data_train))

        random.shuffle(data_train)

        batches = batch_provider(data_train, batch_size, process_batch, report_progress=True)

        rec_loss = 0
        kl_loss = 0

        epoch_start_time = time.time()

        if (epoch + 1) % 8 == 0:
            vae_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        i = 0
        for x in batches:
            vae.train()
            vae.zero_grad()
            rec,var, mu, logvar = vae(x)
            loss_re = loss_function(rec, var,x, mu, logvar)
            loss_re.backward()
            vae_optimizer.step()
            rec_loss += loss_re.item()


            #############################################

            os.makedirs('results_rec', exist_ok=True)
            os.makedirs('results_gen', exist_ok=True)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # report losses and save samples each 60 iterations
            m = 60
            i += 1
            if i % m == 0:
                rec_loss /= m
                kl_loss /= m
                print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f, KL loss: %.9f' % (
                    (epoch + 1), train_epoch, per_epoch_ptime, rec_loss, kl_loss))
                rec_loss = 0
                kl_loss = 0
                with torch.no_grad():
                    vae.eval()
                    x_rec,x_var, _, _ = vae(x)
                    x_var=(x_var**0.50)*0.25*3
                    resultsample = torch.cat([x, x_rec]) * 0.5 + 0.5
                    resultsample=torch.cat([resultsample,x_var])
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               'results_rec/QR/sample_' + str(epoch) + "_" + str(i) + '.png')
                    x_rec,x_var = vae.decode(sample1)
                    resultsample = x_rec * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               'results_gen/QR/sample_' + str(epoch) + "_" + str(i) + '.png')

        del batches
        del data_train
    print("Training finish!... save training results")
    torch.save(vae.state_dict(), "VAEmodel.pkl")

if __name__ == '__main__':
    main()
