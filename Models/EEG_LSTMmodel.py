# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 12:12:40 2022

@author: Abdul Qayyum
"""

#%% LSTM model
import torch
import torch.nn as nn
class LSTMmodule(nn.Module):
  def __init__(self,input_size,h1,h2,number_layers=2):
    super().__init__()
    self.input_size=input_size
    self.h1=h1
    self.h2=h2
    self.number_layers=number_layers
    self.lstm1=nn.LSTM(input_size,h1,number_layers,batch_first=True)
    self.lstm2=nn.LSTM(h1,h2,number_layers,batch_first=True)
    self.fc1=nn.Linear(h2,120)
    self.dropout = nn.Dropout(0.2)
    self.fc2=nn.Linear(120,2)

  def forward(self,x):
    c01,h01=torch.zeros(self.number_layers,x.size(0),self.h1),torch.zeros(self.number_layers,x.size(0),self.h1)
    c02,h02=torch.zeros(self.number_layers,x.size(0),self.h2),torch.zeros(self.number_layers,x.size(0),self.h2)

    out1,(c1,h1)=self.lstm1(x,(c01,h01))
    out2,(c2,h2)=self.lstm2(out1,(c02,h02))
    #print(out2[:,-1,:].shape) # batch_size,hidden_out
    x=self.dropout(self.fc1(out2[:,-1,:]))
    x=self.fc2(x)
    return x
model=LSTMmodule(10,20,10,2)
inpt=torch.randn(5,7,10) # batch_size,seq_len=7,input_size=10
out=model(inpt)
print(out.shape)