# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:54:51 2022

@author: Abdul Qayyum
"""

#%% 2D EEG dataset spectrogram for image classification
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
#import Image
#from skimage import io
# dataset for speech EEG signal


class SpeechEEGdata(Dataset):
    def __init__(self,root,class_1,class_2,is_train):
        self.root=root
        self.class_1=class_1
        self.class_2=class_2
        self.is_train = is_train
        #self.augment_pool = augment_pool()
        
        ######### speech dataset lists #################
        self.pathc=os.path.join(self.root,self.class_1)
        self.pathlistc=os.listdir(self.pathc)
        self.clas1control=[]
        for lstcont in self.pathlistc:
            pathf=os.path.join(self.pathc,lstcont)
            self.clas1control.append((pathf,0))
            
        self.pathc2=os.path.join(self.root,self.class_2)
        self.pathlistc2=os.listdir(self.pathc2)
        self.clas1mdd=[]
        for lstmdd in self.pathlistc2:
            pathfm=os.path.join(self.pathc2,lstmdd)
            self.clas1mdd.append((pathfm,1))
        # total list for class1 and class2
        self.totpath=self.clas1control+self.clas1mdd
        
    
        # for training
        if self.is_train:
            self.transform=transforms.Compose([transforms.ToPILImage(),
                                               transforms.Resize((224,224)),
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.RandomVerticalFlip(p=.05),
                                               transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                ])
        # for validation
        if not self.is_train:
            self.transform=transforms.Compose([transforms.ToPILImage(),
                                               transforms.Resize((224,224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                ])
            
            
    def __getitem__(self,idx):
        
        ################### Speech Dataset spectrum #################
        pathd,label=self.totpath[idx]
        #print(pathd)
        img=cv2.imread(pathd)
        img=self.transform(img)
        
        return {"im1":img,
                "labl1":label}
    
    
    def __len__(self):
        return len(self.totpath)
    
train_data2="D:\\MICCAI2021\\Depression\\Depression\\save_audiao\\trainingdata\\train"    
valid_data2="D:\\MICCAI2021\\Depression\\Depression\\save_audiao\\trainingdata\\val"

dataset_train=SpeechEEGdata(train_data2,'Control','MDD',True)
dataset_valid=SpeechEEGdata(valid_data2,'Control','MDD',False)
len(dataset_train)
len(dataset_valid)
# img,labl=dataset[0]
# print(img.shape)
# print(labl)
# data=dataset_train[0]
# imgen=data['imge']
# imgen.shape
# data['imge']
train_loader=DataLoader(dataset_train,batch_size=4,shuffle=True)
valid_loader=DataLoader(dataset_valid,batch_size=4,shuffle=False) 
len(train_loader.dataset)
len(valid_loader.dataset)