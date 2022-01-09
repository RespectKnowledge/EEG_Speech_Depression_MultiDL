# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:27:14 2022

@author: Abdul Qayyum
"""

import os
import pandas as pd
import scipy.io as sio
import re 
import numpy as np
#pathcsv="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\subjects_information_EEG_128channels_resting_lanzhou_2015_AQ.csv"
datapath="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\EEG_128channels_resting_lanzhou_2015"
# df=pd.read_csv(pathcsv)
# clas_mdd=df[df['type']=='MDD']
# clas_normal=df[df['type']=='HC']
# sub_mdd=clas_mdd['subject id']
# sub_normal=clas_normal['subject id']
# for ii in sub_mdd:
#     pathmdd=os.path.join(datapath,str(ii))
#     print(pathmdd)
################################ MDD class dataset ###################
save_path='D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\Numpydataset\\MDD'
lstpath=os.listdir(datapath)
patren='0201'
mdd_list=list(filter(lambda x: patren in x,lstpath))
dataff11=[]
clasMDD=[]
for mdd in mdd_list:
    filepath=os.path.join(datapath,mdd)
    #print(filepath)
    datafile=sio.loadmat(filepath)
    r = re.compile(".*a0.*")
    dataff=list(filter(r.match, datafile))
    dataff1=dataff[0]
    dataff11.append(dataff1)
    #print(dataff1)
    dataeeg=datafile[dataff1]
    np.save(os.path.join(save_path,dataff1[1:9]+".npy"),dataeeg)
    #print(dataeeg.shape)
    #clasMDD.append((dataeeg,1))
    
save_path1='D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\Numpydataset\\Control'
################ control class dataset #######################
pattern2='0202'
control_lst=list(filter(lambda x: pattern2 in x,lstpath))
pattern3='0203'
control_lst1=list(filter(lambda x: pattern3 in x,lstpath))
contotal=control_lst+control_lst1
dataff22=[]
clcControl=[]
for cntrol in contotal:
    pathc=os.path.join(datapath,cntrol)
    #print(pathc)
    datafilec=sio.loadmat(pathc)
    rc= re.compile(".*a0.*")
    dataffc2=list(filter(rc.match, datafilec))
    dataffc22=dataffc2[0]
    dataff22.append(dataffc22)
    print(dataffc22)
    dataeeg_c=datafilec[dataffc22]
    print(dataeeg_c.shape)
    np.save(os.path.join(save_path1,dataffc22[1:9]+".npy"),dataeeg_c)
    #clcControl.append((dataeeg_c,0))