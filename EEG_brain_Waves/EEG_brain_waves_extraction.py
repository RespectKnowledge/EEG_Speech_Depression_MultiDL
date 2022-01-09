# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:49:49 2022

@author: Abdul Qayyum
"""

#%% EEG dataset brain waves acqustion
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import librosa
import pywt
def dwtBand(signal):
    (cA1, cD1) = pywt.dwt(signal, 'db2', 'smooth')
    (cA2, cD2) = pywt.dwt(cA1, 'db2', 'smooth')
    (cA3, cD3) = pywt.dwt(cA2, 'db2', 'smooth')
    (cA4, cD4) = pywt.dwt(cA3, 'db2', 'smooth')
    (cA5, cD5) = pywt.dwt(cA4, 'db2', 'smooth')
    coefficients_level1 = [cA1, cD1]
    coefficients_level2 = [cA2, cD2, cD1]
    coefficients_level3 = [cA3, cD3, cD2, cD1]
    coefficients_level4 = [cA4, cD4, cD3, cD2, cD1]
    coefficients_level5 = [cA5, cD5, cD4, cD3, cD2, cD1]
    return cA2,cA3,cA4,cA5
path="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\Numpydataset\\"
control=os.path.join(path,"Control")
listfile=os.listdir(control)
ch=[3,4,9,11,12,22,24,28,33,34,36,37,45,52,58,62,70,75,83,92,94,96,97,104,108,116,117,122,124]
save_path="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\newEEdataset\\waves\\Normal_waves\\Normal_Gamma"
save_path1="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\newEEdataset\\waves\\Normal_waves\\Normal_Beta"
save_path2="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\newEEdataset\\waves\\Normal_waves\\Normal_Alpha"
save_path3="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\newEEdataset\\waves\\Normal_waves\\Normal_Delta"
def brainwaves_spectrum(datapath,savepath,condition):
    for i in listfile:
        fpath=os.path.join(control,i)
        arrynp=np.load(fpath)
        npfile=arrynp[:128,:]
        for jj in ch:
            signlfile=npfile[jj]
            cA2,cA3,cA4,cA5=dwtBand(signlfile)
            Gamma=cA2
            Beta=cA3
            Alpha=cA4
            Delta=cA5
            if condition=="Gamma":
                Gamma=cA2
                window_size = 1024
                window = np.hanning(window_size)
                stft  = librosa.core.spectrum.stft(Gamma, n_fft=window_size, hop_length=512, window=window)
                out = 2 * np.abs(stft) / np.sum(window)
                from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
                fig = plt.Figure()
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max),ax=ax,y_axis='log', x_axis='time')   
                fig.savefig(os.path.join(save_path,str(i.split(".")[0])+'_'+str(jj)+'.png'))
            elif condition=="Beta":
                Beta=cA3
                window_size = 1024
                window = np.hanning(window_size)
                stft  = librosa.core.spectrum.stft(Beta, n_fft=window_size, hop_length=512, window=window)
                out = 2 * np.abs(stft) / np.sum(window)
                from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
                fig = plt.Figure()
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max),ax=ax,y_axis='log', x_axis='time')   
                fig.savefig(os.path.join(save_path1,str(i.split(".")[0])+'_'+str(jj)+'.png'))
            elif condition=="Alpha":
                Alpha=cA4
                window_size = 1024
                window = np.hanning(window_size)
                stft  = librosa.core.spectrum.stft(Alpha, n_fft=window_size, hop_length=512, window=window)
                out = 2 * np.abs(stft) / np.sum(window)
                from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
                fig = plt.Figure()
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max),ax=ax,y_axis='log', x_axis='time')   
                fig.savefig(os.path.join(save_path2,str(i.split(".")[0])+'_'+str(jj)+'.png'))
            elif condition=="Delta":
                Delta=cA5
                window_size = 1024
                window = np.hanning(window_size)
                stft  = librosa.core.spectrum.stft(Delta, n_fft=window_size, hop_length=512, window=window)
                out = 2 * np.abs(stft) / np.sum(window)
                from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
                fig = plt.Figure()
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max),ax=ax,y_axis='log', x_axis='time')   
                fig.savefig(os.path.join(save_path3,str(i.split(".")[0])+'_'+str(jj)+'.png'))
            else:
                print("wrong file path")
                
brainwaves_spectrum(listfile,savepath=save_path,condition="Gamma")
brainwaves_spectrum(listfile,savepath=save_path1,condition="Beta")
brainwaves_spectrum(listfile,savepath=save_path2,condition="Alpha")
brainwaves_spectrum(listfile,savepath=save_path3,condition="Delta")

#%% MDD waves spectram extraction for classification
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import librosa
import pywt
def dwtBand(signal):
    (cA1, cD1) = pywt.dwt(signal, 'db2', 'smooth')
    (cA2, cD2) = pywt.dwt(cA1, 'db2', 'smooth')
    (cA3, cD3) = pywt.dwt(cA2, 'db2', 'smooth')
    (cA4, cD4) = pywt.dwt(cA3, 'db2', 'smooth')
    (cA5, cD5) = pywt.dwt(cA4, 'db2', 'smooth')
    coefficients_level1 = [cA1, cD1]
    coefficients_level2 = [cA2, cD2, cD1]
    coefficients_level3 = [cA3, cD3, cD2, cD1]
    coefficients_level4 = [cA4, cD4, cD3, cD2, cD1]
    coefficients_level5 = [cA5, cD5, cD4, cD3, cD2, cD1]
    return cA2,cA3,cA4,cA5
path="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\Numpydataset\\"
control=os.path.join(path,"MDD")
listfile=os.listdir(control)
ch=[3,4,9,11,12,22,24,28,33,34,36,37,45,52,58,62,70,75,83,92,94,96,97,104,108,116,117,122,124]
save_path="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\newEEdataset\\waves\\MDD_waves\\MDD_Gamma"
save_path1="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\newEEdataset\\waves\\MDD_waves\\MDD_Beta"
save_path2="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\newEEdataset\\waves\\MDD_waves\\MDD_Alpha"
save_path3="D:\\MICCAI2021\\Depression\\Depression\\EEG_128channels_resting_lanzhou_2015\\newEEdataset\\waves\\MDD_waves\\MDD_Delta"
def brainwaves_spectrum(datapath,savepath,condition):
    for i in listfile:
        fpath=os.path.join(control,i)
        arrynp=np.load(fpath)
        npfile=arrynp[:128,:]
        for jj in ch:
            signlfile=npfile[jj]
            cA2,cA3,cA4,cA5=dwtBand(signlfile)
            Gamma=cA2
            Beta=cA3
            Alpha=cA4
            Delta=cA5
            if condition=="Gamma":
                Gamma=cA2
                window_size = 1024
                window = np.hanning(window_size)
                stft  = librosa.core.spectrum.stft(Gamma, n_fft=window_size, hop_length=512, window=window)
                out = 2 * np.abs(stft) / np.sum(window)
                from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
                fig = plt.Figure()
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max),ax=ax,y_axis='log', x_axis='time')   
                fig.savefig(os.path.join(save_path,str(i.split(".")[0])+'_'+str(jj)+'.png'))
            elif condition=="Beta":
                Beta=cA3
                window_size = 1024
                window = np.hanning(window_size)
                stft  = librosa.core.spectrum.stft(Beta, n_fft=window_size, hop_length=512, window=window)
                out = 2 * np.abs(stft) / np.sum(window)
                from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
                fig = plt.Figure()
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max),ax=ax,y_axis='log', x_axis='time')   
                fig.savefig(os.path.join(save_path1,str(i.split(".")[0])+'_'+str(jj)+'.png'))
            elif condition=="Alpha":
                Alpha=cA4
                window_size = 1024
                window = np.hanning(window_size)
                stft  = librosa.core.spectrum.stft(Alpha, n_fft=window_size, hop_length=512, window=window)
                out = 2 * np.abs(stft) / np.sum(window)
                from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
                fig = plt.Figure()
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max),ax=ax,y_axis='log', x_axis='time')   
                fig.savefig(os.path.join(save_path2,str(i.split(".")[0])+'_'+str(jj)+'.png'))
            elif condition=="Delta":
                Delta=cA5
                window_size = 1024
                window = np.hanning(window_size)
                stft  = librosa.core.spectrum.stft(Delta, n_fft=window_size, hop_length=512, window=window)
                out = 2 * np.abs(stft) / np.sum(window)
                from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
                fig = plt.Figure()
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max),ax=ax,y_axis='log', x_axis='time')   
                fig.savefig(os.path.join(save_path3,str(i.split(".")[0])+'_'+str(jj)+'.png'))
            else:
                print("wrong file path")
                
brainwaves_spectrum(listfile,savepath=save_path,condition="Gamma")
brainwaves_spectrum(listfile,savepath=save_path1,condition="Beta")
brainwaves_spectrum(listfile,savepath=save_path2,condition="Alpha")
brainwaves_spectrum(listfile,savepath=save_path3,condition="Delta")