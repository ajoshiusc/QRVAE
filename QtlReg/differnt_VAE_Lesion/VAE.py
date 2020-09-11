
#Full assembly of the parts to form the complete network#
#reference:https://github.com/milesial/Pytorch-UNet/blob/master/unet#
#changes number of layyers to 3 instead of 4

import torch.nn.functional as F

from VAE_parts import *
import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        self.down3 = Down(256, 128 )
        self.muC=zConv( 128 , 128 )
        self.logvarC=zConv( 128 , 128)
        self.up1 = Up(128, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64 , bilinear)
        self.outc = OutConv(64, n_classes)

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sparse_loss(self):
        loss = 0
        for class_obj in self.modules():
            if isinstance(class_obj, Up):
                for module_up in class_obj.modules():
                    if isinstance(module_up, nn.Conv2d):
                        loss += torch.mean((module_up.weight.data.clone()) ** 2)
                #for j in range(len(model_children[i])):
            #values = F.relu((model_children[i](values)))
                #loss += torch.mean((values)**2)
                #loss=0
        return loss

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        mu=self.muC(x4)
        logvar=self.logvarC(x4)
        z = self.reparameterize(mu, logvar)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        x = self.up1(z)
        x = self.up2(x)
        x = self.up3(x)
        Q1= self.outc(x)
        
        return mu,logvar, Q1

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 )
        self.muC=zConv(512,512)
        self.logvarC=zConv(512,512)
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64 , bilinear)
        self.outc = OutConv(64, n_classes)
        self.outc2 = OutConv(64, n_classes)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sparse_loss(self):
        loss = 0
        for class_obj in self.modules():
            if isinstance(class_obj, Up):
                for module_up in class_obj.modules():
                    if isinstance(module_up, nn.Conv2d):
                        loss += torch.mean((module_up.weight.data.clone()) ** 2)
                #for j in range(len(model_children[i])):
            #values = F.relu((model_children[i](values)))
                #loss += torch.mean((values)**2)
                #loss=0
        return loss

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        mu=self.muC(x4)
        logvar=self.logvarC(x4)
        z = self.reparameterize(mu, logvar)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        x = self.up1(z)
        x = self.up2(x)
        x = self.up3(x)
        Q1= self.outc(x)
        Q2=self.outc2(x)
        return mu,logvar, Q1,Q2