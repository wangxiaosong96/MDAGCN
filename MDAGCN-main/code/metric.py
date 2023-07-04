from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from collections import Counter
from copy import deepcopy
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from scipy import interp


def metrics(y_true, y_pred, is_sigmoid, isprint = True):
    

    
    y_prob = deepcopy(y_pred)
    

    
    y_true_prob = y_true.reshape(-1)
    y_pred_prob = y_pred.reshape(-1)
    

     
    
    fpr, tpr, _ = roc_curve(y_true_prob, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    
    pr, re, _ = precision_recall_curve(y_true_prob, y_pred_prob)
    aupr = auc(re, pr)

    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1) 
    

    accuracy=accuracy_score(y_true, y_pred)
    f1=f1_score(y_true, y_pred, average="macro")
    recall=recall_score(y_true, y_pred, average='macro')
    precision=precision_score(y_true, y_pred, average='macro')
    


    if isprint:
        print('acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}'.format(accuracy, precision, recall, f1, roc_auc,aupr))
   
    

    return  (y_true, y_pred,y_prob), (accuracy, precision, recall, f1, roc_auc,aupr)
    
   
    