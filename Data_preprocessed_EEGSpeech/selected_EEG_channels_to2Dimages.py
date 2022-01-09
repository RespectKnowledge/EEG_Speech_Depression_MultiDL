# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:33:05 2022

@author: Abdul Qayyum
"""

#%% EEG dataset
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import librosa
path="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\Numpydataset\\"
control=os.path.join(path,"Control")
listfile=os.listdir(control)
ch=[3,4,9,11,12,22,24,28,33,34,36,37,45,52,58,62,70,75,83,92,94,96,97,104,108,116,117,122,124]
save_path="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\newEEdataset\\Control_selct"
for i in listfile:
    fpath=os.path.join(control,i)
    arrynp=np.load(fpath)
    npfile=arrynp[:128,:]
    for jj in ch:
        signlfile=npfile[jj]
        print(signlfile.shape)
        window_size = 1024
        window = np.hanning(window_size)
        stft  = librosa.core.spectrum.stft(signlfile, n_fft=window_size, hop_length=512, window=window)
        out = 2 * np.abs(stft) / np.sum(window)
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max),ax=ax,y_axis='log', x_axis='time')   
        fig.savefig(os.path.join(save_path,str(i.split(".")[0])+'_'+str(jj)+'.png'))
############################### MDD #############################       
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import librosa
path="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\Numpydataset\\"
control=os.path.join(path,"MDD")
listfile=os.listdir(control)
save_path="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\newEEdataset\\MDD_selct"
ch=[3,4,9,11,12,22,24,28,33,34,36,37,45,52,58,62,70,75,83,92,94,96,97,104,108,116,117,122,124]
for i in listfile:
    fpath=os.path.join(control,i)
    arrynp=np.load(fpath)
    npfile=arrynp[:128,:]
    for jj in ch:
        signlfile=npfile[jj]
        print(signlfile.shape)
        window_size = 1024
        window = np.hanning(window_size)
        stft  = librosa.core.spectrum.stft(signlfile, n_fft=window_size, hop_length=512, window=window)
        out = 2 * np.abs(stft) / np.sum(window)
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max),ax=ax,y_axis='log', x_axis='time')   
        fig.savefig(os.path.join(save_path,str(i.split(".")[0])+'_'+str(jj)+'.png'))
        