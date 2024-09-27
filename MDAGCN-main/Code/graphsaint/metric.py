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
    
    
    #y_true = np.array(y_true)

    #print('y_true',y_true)
    #print('y_pred',y_pred.shape)
    
    #np.savetxt("C:/Users/Administrator/Desktop/图采样有向图代码/test data/y_true.txt", y_true,fmt='%s', newline='\n')
    #np.savetxt("C:/Users/Administrator/Desktop/图采样有向图代码/test data/y_pred.txt", y_pred,fmt='%s', newline='\n')
    #y_true.to_csv('C:/Users/Administrator/Desktop/图采样有向图代码/test data/y_true.csv',index=False)
    #y_pred.to_csv('C:/Users/Administrator/Desktop/图采样有向图代码/test data/y_pred.csv',index=False)
    #if isprint:
    #    print('y_pred: 0 = {} | 1 = {} | 2 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]), Counter(y_pred)[2])
    #    print('y_true: 0 = {} | 1 = {} | 2 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]), Counter(y_true)[1])   
        
    #roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    #print('roc_auc',roc_auc)
    #
    
    y_prob = deepcopy(y_pred)
    
    #print（'y_prob:',y_prob）
    
    y_true_prob = y_true.reshape(-1)
    y_pred_prob = y_pred.reshape(-1)
    
    #print('y_true',y_true)
    #print('y_pred',y_pred)  
    #print('y_true_prob',y_true_prob) 
    #print('y_pred_prob',y_pred_prob) 
     
    
    fpr, tpr, _ = roc_curve(y_true_prob, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    #print('roc_auc',roc_auc)
    
    pr, re, _ = precision_recall_curve(y_true_prob, y_pred_prob)
    aupr = auc(re, pr)
    #print('aupr',aupr)
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1) 
    
    #print('y_true',y_true)
    #print('y_pred',y_pred)
    #np.savetxt("C:/Users/Administrator/Desktop/图采样有向图代码/test data/参数/y_true.txt", y_true,fmt='%s', newline='\n')
    
    accuracy=accuracy_score(y_true, y_pred)
    f1=f1_score(y_true, y_pred, average="macro")
    recall=recall_score(y_true, y_pred, average='macro')
    precision=precision_score(y_true, y_pred, average='macro')
    

    
    #*********混淆矩阵***********
    #conf_matrix=confusion_matrix(y_true, y_pred)
    #print('conf_matrix:',conf_matrix)
   # n_classes=3
   # for i in range(n_classes):
       # 逐步获取 真阳，假阳，真阴，假阴四个指标，并计算三个参数
   #    ALL = np.sum(conf_matrix)
       # 对角线上是正确预测的
   #    TP = conf_matrix[i, i]
       # 列加和减去正确预测是该类的假阳
   #    FP = np.sum(conf_matrix[:, i]) - TP
       # 行加和减去正确预测是该类的假阴
   #    FN = np.sum(conf_matrix[i, :]) - TP
       # 全部减去前面三个就是真阴
   #    TN = ALL - TP - FP - FN
    #specificity = TN /(TN+FP)
    #print('specificity:',specificity)

    if isprint:
        print('acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}'.format(accuracy, precision, recall, f1, roc_auc,aupr))
   
    
    #return (y_true, y_pred, y_prob), (accuracy, precision, recall, f1, roc_auc,aupr)
    
    return  (y_true, y_pred,y_prob), (accuracy, precision, recall, f1, roc_auc,aupr)
    
   
    
   
    #roc_auc = roc_auc_score(y_true, y_pred)
    #roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    #roc_auc_micro = roc_auc_score(y_true, y_pred, average='micro')
    #print('roc_auc',roc_auc)
    #print('roc_auc_macro',roc_auc)
    #print('roc_auc_micro',roc_auc_micro)
    #print('y_pred',y_pred)
    

    #y_true = np.argmax(y_true, axis=1)
    #y_pred = np.argmax(y_pred, axis=1)
    
    #print('y_true',y_true)
    #print('y_pred',y_pred)       

    #recall=recall_score(y_true, y_pred, average='macro')
    #print('recall_score_macro',recall)



    #y_true = y_true.reshape(-1)
    #y_pred = y_pred.reshape(-1)
    #if not is_sigmoid:
    #    y_true = np.argmax(y_true, axis=1)
    #    y_pred = np.argmax(y_pred, axis=1)
    #    print('y_true',y_true)
    #    print('y_prob',y_pred)
    #else:
    #    y_pred[y_pred > 0.5] = 1
    #    y_pred[y_pred <= 0.5] = 0

    #print('y_true',y_true)
    #print('y_pred',y_pred)    
    
    #cf=confusion_matrix(y_true, y_pred).ravel()
    #cm=confusion_matrix(y_true, y_pred,)
    #print('cm',cm)
    #conf_matrix = pd.DataFrame(cm, index=['-1','0','1'], columns=['-1','0','1']
    


    #acc=accuracy_score(y_true, y_pred)
    #print('accuracy_score',acc)
    
    
       
    
    #f1=f1_score(y_true, y_pred, average="macro")
    #print('f1_score_macro',f1)
    #print('f1_score',f1_score(y_true, y_pred,average=None))
    #print('f1_macro',f1_score(y_true, y_pred, average="macro"))
    
    #recall=recall_score(y_true, y_pred, average='macro')
    #print('recall_score_macro',recall)
    #print('recall_score',recall_score(y_true, y_pred,average=None))
    #print('recall_score_macro',recall_score(y_true, y_pred, average='macro'))   
    

    
    
    #y_pred_prob = y_pred.reshape(-1)   ##把三列数据转换成一列数据
    #print('y_pred_prob',y_pred_prob)
    
    #y_true = y_true.reshape(-1)
    #y_pred = y_pred.reshape(-1)
    #print('y_true',y_true)
    #print('y_pred',y_pred) 
    
    #print('precision_score',precision_score(y_true, y_pred, average=None))
    #pre=precision_score(y_true, y_pred, average='macro')
    #print('precision_score_macro',pre)
    #print('precision_score_macro',precision_score(y_true, y_pred, average='macro'))     
    


    
    #********通过先计算fpr 和tpe代码再计算auc*********
    ###计算每一类的roc
    #n_classes = 3

    #fpr = dict()
    #tpr = dict()
    #roc_auc = dict()    
    #for i in range(n_classes):
    #    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    #    roc_auc[i] = auc(fpr[i], tpr[i])
    #print('roc_auc:',roc_auc)
    
    ##micro
    #fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #print('roc_auc_micro:',roc_auc["micro"])
    
    ###macro
    #all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #mean_tpr = np.zeros_like(all_fpr)
    #for i in range(n_classes):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # 最后取平均值，计算AUC
    #mean_tpr /= n_classes
    #fpr["macro"] = all_fpr
    #tpr["macro"] = mean_tpr
    #roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #print('roc_auc_macro:',roc_auc["macro"])
      
    
    
    
    
    
    #np.savetxt("C:/Users/Administrator/Desktop/图采样有向图代码/test data/参数/calc_f1_y_pred.txt", y_pred,fmt='%s', newline='\n')
    #return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")
    #return f1_score(y_true, y_pred, average="micro"), f1_score(y_true, y_pred, average="macro")

def calc_metrics(y_true, y_pred, is_sigmoid):
#def calc_metrics(y_true, y_prob, is_sigmoid, isprint = True):
    y_pred_prob = y_pred.reshape(-1)
    print('y_pred_prob',y_pred_prob)
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    #if not is_sigmoid:
    #    y_true = np.argmax(y_true, axis=1)
    #    y_pred = np.argmax(y_pred, axis=1)
    #else:
    #    y_pred[y_pred > 0.5] = 1
    #    y_pred[y_pred <= 0.5] = 0
        
    print('y_true',y_true)
    print('y_pred',y_pred)        
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    print('y_true',y_true)
    print('y_pred',y_pred)

    
    prec_reca_f1_supp_report = classification_report(y_true, y_pred, target_names = ['label 0', 'label 1','label 2'])##['label 0', 'label 1','label 2'])
#    prec_reca_f1_supp_report = classification_report(y_true, y_pred, target_names = ['label 0', 'label 1'])
    print('prec_reca_f1_supp_report',prec_reca_f1_supp_report)
    
    
    #tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    #pos_acc = tp/sum(y_true).item()
    #neg_acc = tn/(len(y_pred)-sum(y_pred).item()) # [y_true=0 & y_pred=0] / y_pred=0#


    roc_auc = roc_auc_score(y_true, y_pred_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_pred_prob)
    aupr = auc(reca, prec)
    
#    if isprint:
#        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
#        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
#        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
#        print('acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}|pos_acc={:.4f}|neg_acc={:.4f}|specificity={:.4f}'.format(accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc, specificity))

    #return (y_true, y_pred, y_prob), (accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc,specificity )
    #print('auc={:.4f}|aupr={:.4f}|pos_acc={:.4f}|neg_acc={:.4f}'.format(roc_auc, aupr, pos_acc, neg_acc))
    print('auc={:.4f}|aupr={:.4f}'.format(roc_auc, aupr))
    return prec_reca_f1_supp_report, pos_acc, neg_acc, roc_auc, aupr

def calc_f1(y_true, y_pred, is_sigmoid):
    if not is_sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def calc_metrics(y_true, y_pred, is_sigmoid):
    y_pred_prob = y_pred.reshape(-1)
    
    if not is_sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    prec_reca_f1_supp_report = classification_report(y_true, y_pred, target_names = ['label 0', 'label 1'])
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    pos_acc = tp/sum(y_true).item()
    neg_acc = tn/(len(y_pred)-sum(y_pred).item()) # [y_true=0 & y_pred=0] / y_pred=0

    roc_auc = roc_auc_score(y_true, y_pred_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_pred_prob)
    aupr = auc(reca, prec)

    return prec_reca_f1_supp_report, pos_acc, neg_acc, roc_auc, aupr


def metrics1111(y_true, y_prob, is_sigmoid, isprint = True):

    y_pred = deepcopy(y_prob)
    
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    
    #print(y_true, y_pred)
    #roc_auc = roc_auc_score(y_true, y_prob)
    #print('roc_auc',roc_auc)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    pos_acc = tp / sum(y_true)
    neg_acc = tn / (len(y_pred) - sum(y_pred)) # [y_true=0 & y_pred=0] / y_pred=0
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    specificity = tn /(tn+fp)
    f1 = 2*precision*recall / (precision+recall)
    
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    
    if isprint:
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
        print('acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}|pos_acc={:.4f}|neg_acc={:.4f}|specificity={:.4f}'.format(accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc, specificity))

    return (y_true, y_pred, y_prob), (accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc,specificity )