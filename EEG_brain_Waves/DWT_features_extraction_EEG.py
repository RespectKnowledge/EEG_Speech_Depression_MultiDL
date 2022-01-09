# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:45:12 2022

@author: Administrateur
"""

#%% wavelet features for CNN and LSTM classifiers

import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import librosa
import pywt
import scipy.io as sio
import scipy
from collections import defaultdict, Counter

path="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\Numpydataset\\"
############ extract normal cases ###############
normal=os.path.join(path,"Control")
listfilenormal=os.listdir(normal)
#################3 extract MDD cases #####################
mdd=os.path.join(path,"MDD")
listfilmdd=os.listdir(mdd)

ch=[3,4,9,11,12,22,24,28,33,34,36,37,45,52,58,62,70,75,83,92,94,96,97,104,108,116,117,122,124]

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics

def extract_features(signal):
    waveletname = 'rbio3.1'
    list_coeff = pywt.wavedec(signal, waveletname)
    features_f = []
    features=[]        
    for coeff in list_coeff:
        features += get_features(coeff)
        features_f.append(features)
    
    X=np.array(features_f)
    
    return X

def DWTfeaturedataset(datapath,savepath,cond):
    if cond=="normal":
        for i in listfilenormal:
            fpath=os.path.join(normal,i)
            arrynp=np.load(fpath)
            npfile=arrynp[:128,:]
            for jj in ch:
                signlfile=npfile[jj]
                X=extract_features(signlfile)
                np.save(os.path.join(savepath,str(i.split(".")[0])+'_'+str(jj)+'.npy'),X)
    elif cond=="MDD":
        for i in listfilmdd:
            fpath=os.path.join(mdd,i)
            arrynp=np.load(fpath)
            npfile=arrynp[:128,:]
            for jj in ch:
                signlfile=npfile[jj]
                X=extract_features(signlfile)
                np.save(os.path.join(savepath,str(i.split(".")[0])+'_'+str(jj)+'.npy'),X)
    else:
        print("wrong datapath")
        
savepathn="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\dwtfeatures_dataset\\Normal" 
savepathmd="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\dwtfeatures_dataset\\MDD"     
DWTfeaturedataset(datapath=path,savepath=savepathn,cond="normal")  
DWTfeaturedataset(datapath=path,savepath=savepathmd,cond="MDD")