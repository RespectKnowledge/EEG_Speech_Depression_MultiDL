# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 12:05:50 2022

@author: Abdul Qayyum
"""

#%% EEG model constructions for depression Assesment
import torch
from torch import nn
import torchvision.models as models
class classify_layer(nn.Module):
    def __init__(self,in_features,num_classes):
        super(classify_layer,self).__init__()
        self.classifier=nn.Sequential(nn.Linear(in_features,128),
                                      nn.ReLU(True),
                                      nn.Linear(128,num_classes))
        print(self.classifier)
        
    def forward(self,x):
        x=self.classifier(x)
        return x
    
# triplet Models for multi dataset
class Multimodel(nn.Module):
    
    def __init__(self,in_features,num_classes):
        super(Multimodel,self).__init__()
        model=models.resnet18(pretrained=False)
        model.fc=torch.nn.Sequential()
        self.model=model
        #print(self.model)
        self.fc=nn.Sequential(nn.Linear(512*2,1024),
                              nn.ReLU(True),
                              nn.Linear(1024,512))
        
        self.classif=classify_layer(in_features,num_classes)
    
    def forward(self,x1,x2,x3):
        F1=self.model(x1)
        F2=self.model(x2)
        F3=self.model(x3)
        # pairwise concatenation of features
        F12=torch.cat((F1,F2),dim=1)
        F23=torch.cat((F2,F3),dim=1)
        F13=torch.cat((F1,F3),dim=1)
        
        f12 = self.fc(F12)
        f23 = self.fc(F23)
        f13 = self.fc(F13)
        
        features = torch.cat((f12, f23, f13), dim=1)  
        print(features.shape)
        score=self.classif(features)
        
        return score
#test model
x1=torch.rand(3,3,224,224)
x2=torch.rand(3,3,224,224)
x3=torch.rand(3,3,224,224)
model=Multimodel(1536,2)
score=model(x1,x2,x3)
print(score.shape)