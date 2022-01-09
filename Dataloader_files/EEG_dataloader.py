# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:42:01 2022

@author: Abdul Qayyum
"""

#%% dataloader to process the EEG dataset
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
import numpy as np

class EEGdataset(Dataset):
    def __init__(self,root,class_1,class_2,rangestart,rangend):
        self.root=root
        self.class_1=class_1
        self.class_2=class_2
        self.rangend=rangend
        self.rangestart=rangestart
    
        ################# Normal class dataset ##############
        self.pathc=os.path.join(self.root,self.class_1)
        self.pathlistc=os.listdir(self.pathc)
        self.clas1contro21=[]
        for lstcont in self.pathlistc:
            pathf=os.path.join(self.pathc,lstcont)
            self.clas1contro21.append((pathf,0))
        
        ################ MDD class ##################
        self.pathc2=os.path.join(self.root,self.class_2)
        self.pathlistc2=os.listdir(self.pathc2)
        self.clas1mdd21=[]
        for lstmdd in self.pathlistc2:
            pathfm=os.path.join(self.pathc2,lstmdd)
            self.clas1mdd21.append((pathfm,1))
            
        self.totpath=self.clas1contro21+self.clas1mdd21
            
    def __getitem__(self,idx):
        ########################## EEG signal dataset ##############
        pathd,label= self.totpath[idx]
        #print(pathd)
        signalEEG=np.load(pathd)
        data = signalEEG.astype(np.float32)
        data=data[:,self.rangestart:self.rangend]
        signalEEG= torch.from_numpy(data)
        #img2=self.transform(img)
        
        return {"im1":signalEEG,
                "labl1":label}
    
    
    def __len__(self):
        return len(self.totpath)
    

train_data3="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\newEEdataset\\trainingdatanew_EEG\\train"    
valid_data3="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\newEEdataset\\trainingdatanew_EEG\\val"

# start and end time to set the range or segment of EEG dataset
rangestart=512
rangend=1024
dataset_train=EEGdataset(train_data3,'Control','MDD',rangestart,rangend)
dataset_valid=EEGdataset(valid_data3,'Control','MDD',rangestart,rangend)
len(dataset_train)
len(dataset_valid)
# img,labl=dataset[0]
# print(img.shape)
# print(labl)
# data=dataset_train[0]
# imgen=data['imge']
# imgen.shape
# data['imge']
train_loader=DataLoader(dataset_train,batch_size=16,shuffle=True)
valid_loader=DataLoader(dataset_valid,batch_size=16,shuffle=False) 
len(train_loader.dataset)
len(valid_loader.dataset)