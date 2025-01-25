# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:10:54 2022

@author: Administrator
"""
import tensorflow as tf

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.losses import binary_crossentropy
from keras.metrics import *
from keras import callbacks
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, confusion_matrix
import numpy as np
import sklearn.metrics as metrics
from collections import Counter
import pandas as pd
import random
from sklearn.model_selection import KFold
from tqdm import tqdm
import scipy.sparse as sp
from copy import deepcopy
import warnings 
import os
from sklearn import preprocessing

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
cf=[[1], [-1], [0], [1],[0],[1]]
print(type(cf))
print(mlb.fit_transform(cf) )









d = {'3': 1, '4': -1, '7': 0,'8':1}

dd= {int(k):v for k,v in d.items()}
print('dd',dd)

class_arr = np.zeros((9, 3))
for key, value in dd.items():
    class_arr[key][value]=1
#print(class_arr)
print(type(class_arr))

            
            
  
 
    

    
    


    
      
         
                  
     


#for key,value in dd.items():#当两个参数时
#     class_arr[key][value]=1

     
     
     
#print(class_arr)
#print(type(class_arr))



