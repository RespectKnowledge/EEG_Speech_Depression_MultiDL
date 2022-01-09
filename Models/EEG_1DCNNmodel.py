# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 12:10:53 2022

@author: Abdul Qayyum
"""

#%% define model for 1DCNN

import torch
import torch.nn as nn
class DCNNEEG(nn.Module):
  def __init__(self,input_size,number_classes=2):
    super().__init__()
    self.input_size=input_size
    self.c1=nn.Conv1d(in_channels=input_size,out_channels=10,kernel_size=5) # 180+1-5=176
    self.b1=nn.BatchNorm1d(10)
    self.mp1=nn.MaxPool1d(2) #176/2=88
    self.c2=nn.Conv1d(10,out_channels=20,kernel_size=3) #88+1-3=86
    self.b2=nn.BatchNorm1d(20)
    self.mp2=nn.MaxPool1d(2) #86/2=43
    
    self.c3=nn.Conv1d(20,out_channels=20,kernel_size=3) #43+1-3=39
    self.b3=nn.BatchNorm1d(20)
    self.mp3=nn.MaxPool1d(3) #39/3=13
    self.fc1=nn.Linear(13*20,128)
    self.fc2=nn.Linear(128,number_classes)

  def forward(self,x):
    x=self.mp1(self.b1(self.c1(x)))
    x=self.mp2(self.b2(self.c2(x)))
    x=self.mp3(self.b3(self.c3(x)))
    x=x.view(-1,13*20)
    x=self.fc1(x)
    x=self.fc2(x)
    
    return x
model=DCNNEEG(15,2) # inp,number_layers,hidden_dim
inpt=torch.randn(10,15,180) # batch_sizexfeatures or input channels xseq_len in time 
out=model(inpt)   
print(out.shape)