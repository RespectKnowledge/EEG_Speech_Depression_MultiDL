# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 06:36:35 2023

@author: aq22
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:02:06 2021

@author: Abdul Qayyum
"""

#%% multtimodel used pretrained transformer to extract pairwise features
# in order to install transformer, pleas install the following library
# pip install timm # you can download the variety of image transformer that are trained 
# on different large datasets.
######################################### multimodel ###################################
import timm
print("Available Vision Transformer Models: ")
print(timm.list_models("vit*"))
#################################################### transformer model #############################
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dataset

class classify_layer(nn.Module):
    def __init__(self,in_features,num_classes):
        super(classify_layer,self).__init__()
        self.classifier=nn.Sequential(nn.Linear(in_features,768),
                                      nn.ReLU(True),
                                      nn.Linear(768,num_classes))
        print(self.classifier)
        
    def forward(self,x):
        x=self.classifier(x)
        return x
#################################################### transformer model #############################
class Mutltimodel_DEP(nn.Module):
    def __init__(self, n_classes, pretrained=False):

        super(Mutltimodel_DEP, self).__init__()

        self.model1 = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
        #self.model.head = nn.Linear(self.model.head.in_features, n_classes)
        self.model1.head= nn.Sequential(nn.Linear(self.model1.head.in_features,self.model1.head.in_features),
                                                  )
        
        self.model2 = timm.create_model("vit_base_patch16_224_in21k", pretrained=pretrained)
        #self.model.head = nn.Linear(self.model.head.in_features, n_classes)
        self.model2.head= nn.Sequential(nn.Linear(self.model2.head.in_features,self.model2.head.in_features),
                                                  )
        
        self.fc=nn.Sequential(nn.Linear(3072,768),
                              nn.Dropout(0.3),
                              nn.ReLU(True),
                              )

        self.classify=classify_layer(768,4)


    def forward(self, x):
        F1=self.model1(x)
        F2=self.model2(x)
        # pairwise concatenation
        Concat1=torch.cat((F1,F2),dim=1) # 768+768=512
        Concate2=torch.cat((F2,F1),dim=1) # 768+768=512
        print(Concat1.shape)
        print(Concate2.shape)
        #features = torch.cat((F1, F2), dim=1)  # 768+768=1024
        features = torch.cat((Concat1, Concate2), dim=1)
        #print(features.shape)
        features=self.fc(features)
        score=self.classify(features)
        return score
    
model = Mutltimodel_DEP(n_classes=4, pretrained=True).to(device)
inp=torch.rand(1,3,224,224).to(device)
out=model(inp)
print(out.shape)