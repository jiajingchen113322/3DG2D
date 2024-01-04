import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def cal_cfm(pred,label,true_label_set,ncls):
    pred=pred.cpu().detach().numpy()
    label=label.cpu().detach().numpy()
    pred=np.argmax(pred,1)

    real_pred=true_label_set[pred]
    real_label=true_label_set[label]

    cfm=confusion_matrix(real_label,real_pred,labels=np.arange(ncls))
    return cfm