# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 12:09:16 2022

@author: Abdul Qayyum
"""

############ model define ##########################
#!pip install efficientnet_pytorch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained("efficientnet-b7")
model._fc = nn.Sequential(nn.Linear(2560, 256),
                                  nn.Dropout(0.5),
                                  nn.ReLU(True),
                                  nn.Linear(256,2),
                                  )
#device = "cuda" if torch.cuda.is_available() else "cpu"
model