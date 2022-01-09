# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:19:53 2022

@author: Abdul Qayyum
"""

#%% Depression analysis using speech and brain signals
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import librosa
path="D:\\MICCAI2021\\Depression\\Depression\\audio_lanzhou_2015\\newwaves\\audio_lanzhou_2015\\"
#pathlist=os.listdir(path)
path_data="D:\\MICCAI2021\\Depression\\Depression\\audio_lanzhou_2015\\subjects_information_audio_lanzhou_2015_AQ.csv"

df=pd.read_csv(path_data)
mdd_class=df.loc[df['type']=='MDD']
normal_class=df.loc[df['type']=='HC']

mdd_sub=mdd_class["subject id"]
normal_sub=normal_class["subject id"]
save_path="D:\\MICCAI2021\\Depression\\Depression\\save_audiao\\MDD"
# ######################## MDD class speech signals ###################
for ii in mdd_sub:
    fpath=os.path.join(path,'0'+str(ii))
    print(fpath)
    wavefile=os.listdir(fpath)
    for file in wavefile:
        pathfile=os.path.join(fpath,str(file))
        print(pathfile)
        audio_path=pathfile 
        y, sr = librosa.load(audio_path, sr=None,mono=True)
        print(f"Sample rate : {sr}")
        # trim silent edges
        audio, _ = librosa.effects.trim(y)
        n_fft = 2048
        hop_length = 256
        n_mels = 128
        S = librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        # For plotting headlessly
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, ax=ax, x_axis='time', y_axis='mel');
        fig.savefig(os.path.join(save_path,str(ii)+'_'+file.split('.')[0]+'.png'))

############################ control class speech signals ##############################
save_path="D:\\MICCAI2021\\Depression\\Depression\\save_audiao\\Control"
for ii in normal_sub:
    fpath=os.path.join(path,'0'+str(ii))
    print(fpath)
    wavefile=os.listdir(fpath)
    for file in wavefile:
        pathfile=os.path.join(fpath,str(file))
        print(pathfile)
        audio_path=pathfile 
        y, sr = librosa.load(audio_path, sr=None,mono=True)
        print(f"Sample rate : {sr}")
        # trim silent edges
        audio, _ = librosa.effects.trim(y)
        n_fft = 2048
        hop_length = 256
        n_mels = 128
        S = librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        # For plotting headlessly
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, ax=ax, x_axis='time', y_axis='mel');
        fig.savefig(os.path.join(save_path,str(ii)+'_'+file.split('.')[0]+'.png'))