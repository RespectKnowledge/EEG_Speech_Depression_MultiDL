# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:59:53 2022

@author: Abdul Qayyum
"""

#%% Argmax
import numpy as np
import os
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from numpy import argmax
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve

pathresults="D:\\MICCAI2021\\Depression\\Results\\Prediciton_Results"

#     return dataparam
dataparam={"Accuracy":[],
           "Percision":[],
           "Recall":[],
           "F1_score":[],
           "Class_1_Accu":[],
           "Class_2_Accu":[],
           }
def performance(path,condp,cong):
    pathG=os.path.join(pathresults,condp)
    pathP=os.path.join(pathresults,cong)
    GT=np.load(pathG)
    Pred=np.load(pathP)
    Pred_arg=np.argmax(Pred,axis=1)
    y_true=GT
    y_pred=Pred_arg
    precision,recall,fscore,support=score(y_true,y_pred,average='macro')
    acc=accuracy_score(y_true, y_pred)
    conf_mat=confusion_matrix(y_true, y_pred)
    class_accuracy=conf_mat.diagonal()/conf_mat.sum(1)
    class1=class_accuracy[0]
    class2=class_accuracy[1]
    dataparam["Accuracy"].append(acc)
    dataparam["Percision"].append(precision)
    dataparam["Recall"].append(recall)
    dataparam["F1_score"].append(fscore)
    dataparam["Class_1_Accu"].append(class_accuracy[0])
    dataparam["Class_2_Accu"].append(class_accuracy[1])
    return dataparam
    
#D=os.path.join(pathresults,"GT_DensNet.npy","P_DensNet.npy"))
performance(pathresults,"GT_DensNet.npy","P_DensNet.npy")
performance(pathresults,"GT_EFNetV1.npy","P_EFNetV1.npy")
performance(pathresults,"GT_effnetv2m.npy","P_effnetv2m.npy") 
performance(pathresults,"GT_effnetv2s.npy","P_effnetv2s.npy") 
performance(pathresults,"GT_SENet.npy","P_SENet.npy") 
performance(pathresults,"GT_mobilNet.npy","P_mobilNet.npy")    

df=pd.DataFrame.from_dict(dataparam)
result=df.rename(index={0: "DensNet", 
                        1: "EFNetV1", 
                        2: "effnetv2m",
                        3: "effnetv2s",
                        4: "SENet",
                        5: "mobilNe"})

result1=df.rename(index={0: "Proposed", 
                        1: "DensNet", 
                        2: "effnetv2m",
                        3: "effnetv2s",
                        4: "SENet",
                        5: "mobilNe"})
result1.to_csv("EEG_Resultsfile.csv")
############################## ROC Measuremnet ######################

def performanceROC(path,condp,cong):
    pathG=os.path.join(pathresults,condp)
    pathP=os.path.join(pathresults,cong)
    GT=np.load(pathG)
    Pred=np.load(pathP)
    encoded_Gt = to_categorical(GT)
    #Pred_arg=np.argmax(Pred,axis=1)
    y_true=encoded_Gt.ravel()
    
    y_pred=Pred.ravel()
    return y_true,y_pred



y_true1,y_pred1=performanceROC(pathresults,"GT_DensNet.npy","P_DensNet.npy")
y_true2,y_pred2=performanceROC(pathresults,"GT_EFNetV1.npy","P_EFNetV1.npy")
y_true3,y_pred3=performanceROC(pathresults,"GT_effnetv2m.npy","P_effnetv2m.npy") 
y_true4,y_pred4=performanceROC(pathresults,"GT_effnetv2s.npy","P_effnetv2s.npy") 
y_true5,y_pred5=performanceROC(pathresults,"GT_SENet.npy","P_SENet.npy") 
y_true6,y_pred6=performanceROC(pathresults,"GT_mobilNet.npy","P_mobilNet.npy")

def rocp(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    au = auc(fpr, tpr)
    return fpr, tpr, thresholds,au

fpr1, tpr1, thresholds1,auc1=rocp(y_true1, y_pred1)
fpr2, tpr2, thresholds2,auc2=rocp(y_true2, y_pred2)
fpr3, tpr3, thresholds3,auc3=rocp(y_true3, y_pred3)
fpr4, tpr4, thresholds4,auc4=rocp(y_true4, y_pred4)
fpr5, tpr5, thresholds5,auc5=rocp(y_true5, y_pred5)
fpr6, tpr6, thresholds6,auc6=rocp(y_true6, y_pred6)

plt.figure(1)
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(fpr1, tpr1,color='red', linestyle='-', linewidth=4.0,label= 'ProposedDNet')

ax.plot(fpr2, tpr2,color='m',linestyle='--', linewidth=3.0,label= 'DensNet')
ax.plot(fpr3, tpr3,marker='o',color='orange', markersize=3,linestyle= '-', linewidth=3.0,label= 'EfficientNet2_m')
ax.plot(fpr4, tpr4,color='blue', linestyle='--',linewidth=3.0, label= 'EfficientNet2_m')
ax.plot(fpr5, tpr5,color='cyan', linestyle='-', linewidth=3.0,label= 'SENET')

ax.plot(fpr6, tpr6,color='green', linestyle='--',linewidth=3.0, label= 'MobileNet')
################# 
plt.xticks(np.arange(0.0, 1.1, step=0.1),fontsize=12,fontweight='bold')
plt.xlabel('False Positve rate',fontsize=16,fontweight='bold')
plt.yticks(np.arange(0.0, 1.1, step=0.1),fontsize=12,fontweight='bold')
plt.ylabel('True Positive rate',fontsize=16,fontweight='bold')
#plt.title('ROC curve')
plt.title('ROC Curves', fontweight='bold', fontsize=16)
plt.xlim([0,0.2])
#plt.ylim([0,0.99])
plt.legend(prop={'size':13}, loc='lower right',fontsize=16)
#ax.legend(loc="bottom right",fontweight='bold')
plt.savefig('EEGCNN.png',dpi=100)
plt.savefig("EEG.svg")


##################### PRCurves ###############################

def preRecallCurve(y_true,y_pred):
    average_precision = average_precision_score(y_true,y_pred)
    p, r, _ = precision_recall_curve(y_true,y_pred)
    return average_precision,p,r
average_precision1,p1,r1=preRecallCurve(y_true1,y_pred1)
average_precision2,p2,r2=preRecallCurve(y_true2,y_pred2)
average_precision3,p3,r3=preRecallCurve(y_true3,y_pred3)
average_precision4,p4,r4=preRecallCurve(y_true4,y_pred4)
average_precision5,p5,r5=preRecallCurve(y_true5,y_pred5)
average_precision6,p6,r6=preRecallCurve(y_true6,y_pred6)
    
plt.figure(figsize=(10,6))
plt.step(r1, p1,color='brown', linewidth=4, where='post', label='ProposedDNet')
plt.step(r2, p2,color='green', linewidth=3, where='post', label='DensNet')
plt.step(r3, p3, linewidth=3,color='orange', where='post', label='EfficnetNetV2_m')
plt.step(r4, p4,color='pink', linestyle='--',linewidth=3, where='post', label='EfficnetNetV2_s')
plt.step(r5, p5,color='red',linestyle='--', linewidth=3, where='post', label='SENet')
plt.step(r6, r6,color='blue',linestyle='--', linewidth=3, where='post', label='MobileNet')

# plt.ylim([0.0, 1.09])
# plt.xlim([0.0, 1.0])
plt.xlabel('Recall',fontsize=16,fontweight='bold')
plt.ylabel('Precision',fontsize=16,fontweight='bold')
# plt.title('Precision-Recall curve')
# plt.legend(loc="lower left")

plt.xticks(fontsize=12,fontweight='bold')
#plt.xlabel('False Positve rate',fontsize=16,fontweight='bold')
plt.yticks(fontsize=12,fontweight='bold')
#plt.ylabel('True Positive rate',fontsize=16,fontweight='bold')
#plt.title('ROC curve')
plt.title('Precision-Recall curve', fontweight='bold', fontsize=16)
plt.legend(prop={'size':13}, loc='lower left',fontsize=16)
plt.xlim([0,0.99])
#plt.ylim([0,1])
plt.savefig('EEGCNNPR.png',dpi=100)
plt.savefig("EEGPR.svg")
#%% speech waves results
import numpy as np
import os
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from numpy import argmax
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve

pathresults="D:\\MICCAI2021\\Depression\\Results\\audiao\\"

#     return dataparam
dataparam={"Accuracy":[],
           "Percision":[],
           "Recall":[],
           "F1_score":[],
           "Class_1_Accu":[],
           "Class_2_Accu":[],
           }
def performance(path,condp,cong):
    pathG=os.path.join(path,condp)
    pathP=os.path.join(path,cong)
    GT=np.load(pathG)
    Pred=np.load(pathP)
    Pred_arg=np.argmax(Pred,axis=1)
    y_true=GT
    y_pred=Pred_arg
    precision,recall,fscore,support=score(y_true,y_pred,average='macro')
    acc=accuracy_score(y_true, y_pred)
    conf_mat=confusion_matrix(y_true, y_pred)
    class_accuracy=conf_mat.diagonal()/conf_mat.sum(1)
    class1=class_accuracy[0]
    class2=class_accuracy[1]
    dataparam["Accuracy"].append(acc)
    dataparam["Percision"].append(precision)
    dataparam["Recall"].append(recall)
    dataparam["F1_score"].append(fscore)
    dataparam["Class_1_Accu"].append(class_accuracy[0])
    dataparam["Class_2_Accu"].append(class_accuracy[1])
    return dataparam
    
pathresultsd=os.path.join(pathresults,"ResNet18") 
pathresultse=os.path.join(pathresults,"eficieNetV1")
pathresultse1=os.path.join(pathresults,"efficiNetv2")   
pathresultse2=os.path.join(pathresults,"DensNet201")  
pathresultss=os.path.join(pathresults,"Sqeeznet") 
pathresultsv=os.path.join(pathresults,"MobileNetV2") 
pathtest=os.path.join(pathresultsd,"GT_densenet201_audio.npy")
#D=os.path.join(pathresults,"GT_DensNet.npy","P_DensNet.npy"))
performance(pathresultsd,"GT_DensNet.npy","prediciton_DensNet.npy")
performance(pathresultse,"GT_effnetv1_audio.npy","prediciton_effnetv1_audio.npy")
performance(pathresultse1,"GT_effnetv2_audio.npy","prediciton_effnetv2_audio.npy") 
performance(pathresultse2,"GT_densenet201_audio.npy","prediciton_densenet201_audio.npy") 
performance(pathresultss,"GT_squeezenet1_audio.npy","prediciton_squeezenet1_audio.npy") 
performance(pathresultsv,"GT_mobilenet_v2_audio.npy","prediciton_mobilenet_v2_audio.npy")    

df=pd.DataFrame.from_dict(dataparam)
result=df.rename(index={0: "DensNet", 
                        1: "EFNetV1", 
                        2: "effnetv2m",
                        3: "effnetv2s",
                        4: "SENet",
                        5: "mobilNe"})

result1=df.rename(index={0: "Proposed", 
                        1: "DensNet", 
                        2: "effnetv2m",
                        3: "effnetv2s",
                        4: "SENet",
                        5: "mobilNe"})
result1.to_csv("EEGSound_Resultsfile1.csv")
############################## ROC Measuremnet ######################

def performanceROC(path,condp,cong):
    pathG=os.path.join(path,condp)
    pathP=os.path.join(path,cong)
    GT=np.load(pathG)
    Pred=np.load(pathP)
    encoded_Gt = to_categorical(GT)
    #Pred_arg=np.argmax(Pred,axis=1)
    y_true=encoded_Gt.ravel()
    
    y_pred=Pred.ravel()
    return y_true,y_pred



y_true1, y_pred1=performanceROC(pathresultsd,"GT_DensNet.npy","prediciton_DensNet.npy")
y_true2, y_pred2=performanceROC(pathresultse,"GT_effnetv1_audio.npy","prediciton_effnetv1_audio.npy")
y_true3, y_pred3=performanceROC(pathresultse1,"GT_effnetv2_audio.npy","prediciton_effnetv2_audio.npy") 
y_true4, y_pred4=performanceROC(pathresultse2,"GT_densenet201_audio.npy","prediciton_densenet201_audio.npy") 
y_true5, y_pred5=performanceROC(pathresultss,"GT_squeezenet1_audio.npy","prediciton_squeezenet1_audio.npy") 
y_true6, y_pred6=performanceROC(pathresultsv,"GT_mobilenet_v2_audio.npy","prediciton_mobilenet_v2_audio.npy")

def rocp(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    au = auc(fpr, tpr)
    return fpr, tpr, thresholds,au

fpr1, tpr1, thresholds1,auc1=rocp(y_true1, y_pred1)
fpr2, tpr2, thresholds2,auc2=rocp(y_true2, y_pred2)
fpr3, tpr3, thresholds3,auc3=rocp(y_true3, y_pred3)
fpr4, tpr4, thresholds4,auc4=rocp(y_true4, y_pred4)
fpr5, tpr5, thresholds5,auc5=rocp(y_true5, y_pred5)
fpr6, tpr6, thresholds6,auc6=rocp(y_true6, y_pred6)

plt.figure(1)
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(fpr1, tpr1,color='red', linestyle='-', linewidth=4.0,label= 'ProposedDNet')

ax.plot(fpr2, tpr2,color='m',linestyle='--', linewidth=3.0,label= 'DensNet')
ax.plot(fpr3, tpr3,marker='o',color='orange', markersize=3,linestyle= '-', linewidth=3.0,label= 'EfficientNet2_m')
ax.plot(fpr4, tpr4,color='blue', linestyle='--',linewidth=3.0, label= 'EfficientNet2_m')
ax.plot(fpr5, tpr5,color='cyan', linestyle='-', linewidth=3.0,label= 'SENET')

ax.plot(fpr6, tpr6,color='green', linestyle='--',linewidth=3.0, label= 'MobileNet')
################# 
plt.xticks(np.arange(0.0, 1.1, step=0.1),fontsize=12,fontweight='bold')
plt.xlabel('False Positve rate',fontsize=16,fontweight='bold')
plt.yticks(np.arange(0.0, 1.1, step=0.1),fontsize=12,fontweight='bold')
plt.ylabel('True Positive rate',fontsize=16,fontweight='bold')
#plt.title('ROC curve')
plt.title('ROC Curves', fontweight='bold', fontsize=16)
plt.xlim([0,0.2])
#plt.ylim([0,0.99])
plt.legend(prop={'size':13}, loc='lower right',fontsize=16)
#ax.legend(loc="bottom right",fontweight='bold')
plt.savefig('EEGCNNsound.png',dpi=100)
plt.savefig("EEGsound.svg")


##################### PRCurves ###############################

def preRecallCurve(y_true,y_pred):
    average_precision = average_precision_score(y_true,y_pred)
    p, r, _ = precision_recall_curve(y_true,y_pred)
    return average_precision,p,r
average_precision1,p1,r1=preRecallCurve(y_true1,y_pred1)
average_precision2,p2,r2=preRecallCurve(y_true2,y_pred2)
average_precision3,p3,r3=preRecallCurve(y_true3,y_pred3)
average_precision4,p4,r4=preRecallCurve(y_true4,y_pred4)
average_precision5,p5,r5=preRecallCurve(y_true5,y_pred5)
average_precision6,p6,r6=preRecallCurve(y_true6,y_pred6)
    
plt.figure(figsize=(10,6))
plt.step(r1, p1,color='brown', linewidth=4, where='post', label='ProposedDNet')
plt.step(r2, p2,color='green', linewidth=3, where='post', label='DensNet')
plt.step(r3, p3, linewidth=3,color='orange', where='post', label='EfficnetNetV2_m')
plt.step(r4, p4,color='pink', linestyle='--',linewidth=3, where='post', label='EfficnetNetV2_s')
plt.step(r5, p5,color='red',linestyle='--', linewidth=3, where='post', label='SENet')
plt.step(r6, r6,color='blue',linestyle='--', linewidth=3, where='post', label='MobileNet')

# plt.ylim([0.0, 1.09])
# plt.xlim([0.0, 1.0])
plt.xlabel('Recall',fontsize=16,fontweight='bold')
plt.ylabel('Precision',fontsize=16,fontweight='bold')
# plt.title('Precision-Recall curve')
# plt.legend(loc="lower left")

plt.xticks(fontsize=12,fontweight='bold')
#plt.xlabel('False Positve rate',fontsize=16,fontweight='bold')
plt.yticks(fontsize=12,fontweight='bold')
#plt.ylabel('True Positive rate',fontsize=16,fontweight='bold')
#plt.title('ROC curve')
plt.title('Precision-Recall curve', fontweight='bold', fontsize=16)
plt.legend(prop={'size':13}, loc='lower left',fontsize=16)
plt.xlim([0,0.99])
#plt.ylim([0,1])
plt.savefig('EEGCNNPRsound.png',dpi=100)
plt.savefig("EEGPRsound.svg")

#%% Confusion Matrix
import numpy as np
import os
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from numpy import argmax
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve

pathresults="D:\\MICCAI2021\\Depression\\Results\\audiao\\"

#     return dataparam
dataparam={"Accuracy":[],
           "Percision":[],
           "Recall":[],
           "F1_score":[],
           "Class_1_Accu":[],
           "Class_2_Accu":[],
           }
def performanceCM(path,condp,cong):
    pathG=os.path.join(path,condp)
    pathP=os.path.join(path,cong)
    GT=np.load(pathG)
    Pred=np.load(pathP)
    Pred_arg=np.argmax(Pred,axis=1)
    y_true=GT
    y_pred=Pred_arg
    return y_true,y_pred
    
pathresultsd=os.path.join(pathresults,"ResNet18") 
pathresultse=os.path.join(pathresults,"eficieNetV1")
pathresultse1=os.path.join(pathresults,"efficiNetv2")   
pathresultse2=os.path.join(pathresults,"DensNet201")  
pathresultss=os.path.join(pathresults,"Sqeeznet") 
pathresultsv=os.path.join(pathresults,"MobileNetV2") 
pathtest=os.path.join(pathresultsd,"GT_densenet201_audio.npy")
#D=os.path.join(pathresults,"GT_DensNet.npy","P_DensNet.npy"))
y_true1,y_pred1=performanceCM(pathresultsd,"GT_DensNet.npy","prediciton_DensNet.npy")
y_true2,y_pred2=performanceCM(pathresultse,"GT_effnetv1_audio.npy","prediciton_effnetv1_audio.npy")
y_true3,y_pred3=performanceCM(pathresultse1,"GT_effnetv2_audio.npy","prediciton_effnetv2_audio.npy") 
y_true4,y_pred4=performanceCM(pathresultse2,"GT_densenet201_audio.npy","prediciton_densenet201_audio.npy") 
y_true5,y_pred5=performanceCM(pathresultss,"GT_squeezenet1_audio.npy","prediciton_squeezenet1_audio.npy") 
y_true6,y_pred6=performanceCM(pathresultsv,"GT_mobilenet_v2_audio.npy","prediciton_mobilenet_v2_audio.npy")    

classes= ['Normal','Depression']

#Get the confusion matrix
import matplotlib.pyplot as plt
#cm = confusion_matrix(y_test3, y_pred2)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_true1, y_pred1)
import itertools
# fontweight='bold'
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=17)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=13)
    plt.yticks(tick_marks, classes,fontsize=13)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(2)
        #print("Normalized confusion matrix")
    else:
        cm=cm
        #print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=15)

    plt.tight_layout()
    plt.ylabel('True label',fontsize=10)
    plt.xlabel('Predicted label',fontsize=10)


#np.set_printoptions(precision=2)

fig1 = plt.figure(figsize=(4,4))
plot_confusion_matrix(conf_mat, classes=classes, title='Confusion matrix')
#fig1.savefig('../cm_wo_norm.jpg')
#plt.show()
plt.savefig('CMProposed1.png',dpi=100)
plt.savefig("CMProposed1.svg")

# np.set_printoptions(precision=2)

# fig2 = plt.figure(figsize=(7,6))
# plot_confusion_matrix(conf_mat, classes=classes, normalize = True, title='Normalized Confusion matrix')
# fig2.savefig('cm_norm.jpg')
# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPDF, renderPM
# drawing = svg2rlg("Proposed.svg")
# renderPDF.drawToFile(drawing, "file.pdf")
# renderPM.drawToFile(drawing, "file.png", fmt="PNG")

expected = y_true6
predicted = y_pred6
cf =confusion_matrix(expected,predicted)
cf


plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
tick_marks = np.arange(len(set(expected))) # length of classes
class_labels = ['Normal','Depression']
tick_marks
plt.xticks(tick_marks,class_labels,fontsize=10)
plt.yticks(tick_marks,class_labels,fontsize=10)
# plotting text value inside cells
thresh = cf.max() / 2.
for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
    plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
#plt.show();

plt.savefig('CMObileNet.png',dpi=100)
plt.savefig("CMObileNet.svg")

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
drawing = svg2rlg("CMProposed.svg")
renderPDF.drawToFile(drawing, "file1.pdf")
#%% Confusion Matrix EEG
import numpy as np
import os
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from numpy import argmax
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve

pathresults="D:\\MICCAI2021\\Depression\\Results\\Prediciton_Results"


def performanceCM(path,condp,cong):
    pathG=os.path.join(path,condp)
    pathP=os.path.join(path,cong)
    GT=np.load(pathG)
    Pred=np.load(pathP)
    Pred_arg=np.argmax(Pred,axis=1)
    y_true=GT
    y_pred=Pred_arg
    return y_true,y_pred
    
#D=os.path.join(pathresults,"GT_DensNet.npy","P_DensNet.npy"))
y_true1,y_pred1=performanceCM(pathresults,"GT_DensNet.npy","P_DensNet.npy")
y_true2,y_pred2=performanceCM(pathresults,"GT_EFNetV1.npy","P_EFNetV1.npy")
y_true3,y_pred3=performanceCM(pathresults,"GT_effnetv2m.npy","P_effnetv2m.npy") 
y_true4,y_pred4=performanceCM(pathresults,"GT_effnetv2s.npy","P_effnetv2s.npy") 
y_true5,y_pred5=performanceCM(pathresults,"GT_SENet.npy","P_SENet.npy") 
y_true6,y_pred6=performanceCM(pathresults,"GT_mobilNet.npy","P_mobilNet.npy")   

expected = y_true6
predicted = y_pred6
cf =confusion_matrix(expected,predicted)
cf
import itertools

plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
tick_marks = np.arange(len(set(expected))) # length of classes
class_labels = ['Normal','Depression']
tick_marks
plt.xticks(tick_marks,class_labels,fontsize=10)
plt.yticks(tick_marks,class_labels,fontsize=10)
# plotting text value inside cells
thresh = cf.max() / 2.
for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
    plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
#plt.show();

plt.savefig('CMMobileNetEEG.png',dpi=100)
plt.savefig("CMMobileNetEEG.svg")
