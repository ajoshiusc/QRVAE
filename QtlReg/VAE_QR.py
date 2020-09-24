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
from net_QR import *
import numpy as np
import pickle
import time
import random
import os
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from PIL import Image


im_size = 128


def loss_function(recon_x, x, mu, logvar,Q):
    BCE = torch.mean(torch.mean(torch.max(Q * (x-recon_x ), (Q - 1) * (x-recon_x)).view(-1, 128*128*3),(1))) 

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD 


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
            rec, mu, logvar = vae(x)

            loss_re_q1, loss_kl_q1 = loss_function(rec[:,0:3,:,:], x, mu, logvar,0.15)
            loss_re_q2, loss_kl_q2 = loss_function(rec[:,3:7,:,:], x, mu, logvar,0.5)
            loss_re =loss_re_q1+loss_re_q2
            loss_kl=loss_kl_q1+loss_kl_q2
            (loss_re + loss_kl).backward()
            vae_optimizer.step()
            rec_loss += loss_re.item()
            kl_loss += loss_kl.item()

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
                    x_rec, _, _ = vae(x)
                    x_mean=x_rec[:,0:3,:,:]
                    x_std=(x_rec[:,3:7,:,:]-x_rec[:,0:3,:,:])*0.25
                    resultsample = torch.cat([x, x_mean]) * 0.5 + 0.5
                    resultsample=torch.cat([resultsample,x_std])
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               'results_rec/QR/sample_' + str(epoch) + "_" + str(i) + '.png')
                    x_rec = vae.decode(sample1)
                    resultsample = x_rec[:,3:7,:,:] * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               'results_gen/QR/sample_' + str(epoch) + "_" + str(i) + '.png')

        del batches
        del data_train
    print("Training finish!... save training results")
    torch.save(vae.state_dict(), "results_gen/QR/VAEmodel.pkl")

if __name__ == '__main__':
    main()
