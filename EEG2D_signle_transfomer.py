# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 06:31:43 2023

@author: aq22
"""
import torch.nn as nn
import timm
class ViTBase16(nn.Module):
    def __init__(self, model_name, n_classes, pretrained=False):

        super(ViTBase16, self).__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)
        # self.model.head = nn.Sequential(nn.Linear(self.model.head.in_features, 512),
        #                                 nn.Dropout(0.5),
        #                                 nn.ReLU(True),
        #                                 nn.Linear(512,n_classes),
        #                                 )
        #self.model.classifier = nn.Linear(self.model.classifier.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
model_list=['vit_base_patch16_224', 
 'vit_base_patch16_224_in21k', 
 'vit_base_patch16_224_miil', 
 'vit_base_patch16_224_miil_in21k', 
 'vit_base_patch32_224', 
 'vit_base_patch32_224_in21k',  
 'vit_base_r26_s32_224', 
 'vit_base_r50_s16_224', 
 'vit_base_r50_s16_224_in21k',  
 'vit_base_resnet26d_224', 
 'vit_base_resnet50_224_in21k', 
 'vit_base_resnet50d_224']
###n_classes=n_classes
for model_name in model_list:
    print(model_name)
    model = ViTBase16(model_name,n_classes=3, pretrained=True)
    model.cuda()