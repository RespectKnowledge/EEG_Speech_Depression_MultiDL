# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:21:43 2022

@author: Administrateur
"""

#%% SFTF spectrum signal generation
#%Depression analysis using speech and brain signals
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
save_path="D:\\MICCAI2021\\Depression\\Depression\\save_audiao\\SFTF\\MDD"
# # ######################## MDD class speech signals ###################
for ii in mdd_sub:
    fpath=os.path.join(path,'0'+str(ii))
    print(fpath)
    wavefile=os.listdir(fpath)
    for file in wavefile:
        pathfile=os.path.join(fpath,str(file))
        print(pathfile)
        audio_path=pathfile 
        y, sr = librosa.load(pathfile)
        window_size = 1024
        window = np.hanning(window_size)
        stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
        out = 2 * np.abs(stft) / np.sum(window)
        # For plotting headlessly
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')   
        fig.savefig(os.path.join(save_path,str(ii)+'_'+file.split('.')[0]+'.png'))      

save_path="D:\\MICCAI2021\\Depression\\Depression\\save_audiao\\SFTF\\Control"
# ######################## MDD class speech signals ###################
for ii in normal_sub:
    fpath=os.path.join(path,'0'+str(ii))
    print(fpath)
    wavefile=os.listdir(fpath)
    for file in wavefile:
        pathfile=os.path.join(fpath,str(file))
        print(pathfile)
        audio_path=pathfile 
        y, sr = librosa.load(pathfile)
        window_size = 1024
        window = np.hanning(window_size)
        stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
        out = 2 * np.abs(stft) / np.sum(window)
        # For plotting headlessly
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')   
        fig.savefig(os.path.join(save_path,str(ii)+'_'+file.split('.')[0]+'.png'))  
        