# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:40:55 2024

@author: aq22
"""

#%%
import torch
import torch.nn as nn
class CNN_1D(nn.Module):
    def __init__(self):
        super(CNN_1D,self).__init__()
        self.layer1=nn.Conv1d(in_channels=129, out_channels=16, kernel_size=3,stride=1,padding=1)
        self.p1=nn.MaxPool1d(2)
        self.layer2=nn.Conv1d(16, 32,3,1,1)
        self.p2=nn.MaxPool1d(2)
        self.faltten=nn.Flatten()
        self.fc1=nn.Linear(in_features=2048, out_features=128)
        self.fc2=nn.Linear(128,2)
    def forward(self,x):
        x=self.layer1(x)
        print(f'layer1_cnn: {x.shape}')
        x=self.p1(x)
        print(x.shape)
        x=self.layer2(x)
        print(x.shape)
        x=self.p2(x)
        x=self.faltten(x)
        print(x.shape)
        fc1=self.fc1(x)
        print(fc1.shape)
        out=self.fc2(fc1)
        print(out.shape)
        return out
        
model=CNN_1D()
inp=torch.rand(10,129,256)
out=model(inp)
print(out.shape) ## batchxclasses