from graphsaint.globals import *
from graphsaint.pytorch_version.models import GraphSAINT
from graphsaint.pytorch_version.minibatch import Minibatch
from graphsaint.utils import *
from graphsaint.metric import *
from graphsaint.pytorch_version.utils import *
import numpy as np

import torch
import time

import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
def evaluate_full_batch(model, minibatch, mode='Test'):####Test  val
    """
    Full batch evaluation: for validation and test sets only.
        When calculating the F1 score, we will mask the relevant root nodes
        仅用于验证和测试集。在计算F1分数时，我们将对相关的根节点进行掩码   ####根节点：是没有父节点的节点
        (e.g., those belonging to the val / test sets).
    """
    loss,preds,labels = model.eval_step(*minibatch.one_batch(mode=mode))  ###model.eval_step向前传播 得到输出层的结果————————minibatch.one_batch

    if mode == 'val':
        printf('Val: loss = {:.4f}'.format(loss), style = 'red')
        node_target = [minibatch.node_val]
    elif mode == 'Test':
        printf('Test: loss = {:.4f}'.format(los), style = 'red')
        node_target = [minibatch.node_test]
    else:
        print('Validation & Test: ')
        assert mode == 'valtest'
        node_target = [minibatch.node_val, minibatch.node_test]

    accuracy, precision, recall, f1, roc_auc, aupr = [], [], [], [], [], []
    for n in node_target:

        ys, performances = metrics(to_numpy(labels[n,:]), to_numpy(preds[n,:]), model.sigmoid_loss)  #### metric.py代码


        accuracy.append(performances[0])
        precision.append(performances[1])
        recall.append(performances[2])
        f1.append(performances[3])
        roc_auc.append(performances[4])
        aupr.append(performances[5])

    

    roc_auc = roc_auc[0] if len(roc_auc) == 1 else roc_auc
    aupr = aupr[0] if len(aupr) == 1 else aupr
    f1 = f1[0] if len(f1) == 1 else f1
    recall = recall[0] if len(recall) == 1 else recall
    precision = precision[0] if len(precision) == 1 else precision
    accuracy = accuracy[0] if len(accuracy) == 1 else accuracy
    

    
    return loss, ys, (accuracy, precision, recall, f1, roc_auc,aupr) # (y_true, y_pred, y_prob), (accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc)
    #return loss, ys, (accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc)
    


def prepare(train_data,train_params,arch_gcn):
    """
    Prepare some data structure and initialize model / minibatch handler before   
    the actual iterative training taking place.
    """  ##  准备一些数据结构和初始化模型           minibatch处理程序之前实际的迭代训练
    ###初始化数据
    ############minibatch和minibatch_eval的前期的数据准备并进行采样都是一样的，但是在最后的训练迭代时是有所区别的#####
    adj_full, adj_train, feat_full, class_arr,role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    #print('adj_train:',adj_train)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]   ###  m_classes 3  标签的列数 有3列标签

    minibatch = Minibatch(adj_full_norm, adj_train, role, train_params)
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)    #####准备一些数据结构和初始化模型
    printf("TOTAL NUM OF PARAMS = {}".format(sum(p.numel() for p in model.parameters())), style="yellow")##printf()函数是格式化输出函数
    minibatch_eval=Minibatch(adj_full_norm, adj_train, role, train_params, cpu_eval=True)
    model_eval=GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)
    #print('model_eval:',model_eval)
    
    if args_global.gpu >= 0:
        model = model.to(device)

    return model, minibatch, minibatch_eval, model_eval


def train(train_phases, model, minibatch, minibatch_eval, model_eval, eval_val_every):

    if not args_global.cpu_eval:
        minibatch_eval=minibatch
    epoch_ph_start = 0 
    auc_best, ep_best = 0, -1
    time_train = 0
    dir_saver = '{}/pytorch_models'.format(args_global.dir_log)
    path_saver = '{}/pytorch_models/mirna_disease_saved_model_{}.pkl'.format(args_global.dir_log, timestamp)


    for ip, phase in enumerate(train_phases):  #用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        printf('START PHASE {:4d}'.format(ip),style='underline')
        minibatch.set_sampler(phase)##用边采样模型
        num_batches = minibatch.num_training_batches()   ###2
        print('num_batche',num_batches)
        for e in range(epoch_ph_start, int(phase['end'])):    ###epoch  设置为end=1000
            printf('Epoch {:4d}'.format(e),style='bold')  ###规定输出的格式
            minibatch.shuffle()   ####随机序列排序
            
            l_loss_tr, lr_accuracy_tr, lr_precision_tr, lr_recall_tr, lr_f1_tr, lr_roc_auc_tr, lr_aupr_tr, lr_pos_acc_tr, lr_neg_acc_tr = [], [], [], [], [], [], [], [], []
            
            time_train_ep = 0
            while not minibatch.end():
                t1 = time.time()
                loss_train,preds_train,labels_train = model.train_step(*minibatch.one_batch(mode='train'))####生成结果     *是指针，  指针可以任意转换类型，所以字符指针返回局部变量或临时变量的地址
                time_train_ep += time.time() - t1

                if not minibatch.batch_num % args_global.eval_train_every:####  % 计算 a 除以 b 得出的余数。
                    

                    ys_train, metrics_train = metrics(to_numpy(labels_train[:, :]),to_numpy(preds_train[:, :]),model.sigmoid_loss, isprint = True)#, isprint = True
                    print('labels_train',labels_train.shape)

                    l_loss_tr.append(loss_train)
                    lr_accuracy_tr.append(metrics_train[0])
                    lr_precision_tr.append(metrics_train[1])
                    lr_recall_tr.append(metrics_train[2])
                    lr_f1_tr.append(metrics_train[3])
                    lr_roc_auc_tr.append(metrics_train[4])
                    lr_aupr_tr.append(metrics_train[5])

                    
            if (e+1)%eval_val_every == 0:
                if args_global.cpu_eval:
                    torch.save(model.state_dict(),'tmp.pkl')
                    model_eval.load_state_dict(torch.load('tmp.pkl',map_location=lambda storage, loc: storage))
                else:
                    model_eval = model
                    
                printf('Train (Ep avg): loss = {:.4f} | Time = {:.4f}sec'.format(f_mean(l_loss_tr), time_train_ep), style = 'yellow')
                printf('acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}'.format(
                    f_mean(lr_accuracy_tr), f_mean(lr_precision_tr), f_mean(lr_recall_tr), f_mean(lr_f1_tr), f_mean(lr_roc_auc_tr), f_mean(lr_aupr_tr)), 
                    style = 'yellow')
                loss_val, ys_val, metrics_val = evaluate_full_batch(model_eval, minibatch_eval, mode='val')
                
                auc_val = metrics_val[4]
                if auc_val > auc_best:
                    auc_best, ep_best = auc_val, e
                    if not os.path.exists(dir_saver):
                        os.makedirs(dir_saver)
                    printf('  Saving model ...', style='yellow')
                    torch.save(model.state_dict(), path_saver)
            time_train += time_train_ep
        epoch_ph_start = int(phase['end'])
    printf("Optimization Finished!", style="yellow")
    if ep_best >= 0:
        if args_global.cpu_eval:
            model_eval.load_state_dict(torch.load(path_saver, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load(path_saver))
            model_eval=model
        printf('  Restoring model ...', style='yellow')
        
    printf('Best Epoch = ' + str(ep_best), style = 'red')
    loss_test, ys_test, metrics_test = evaluate_full_batch(model_eval, minibatch_eval, mode='val')####验证的ys_test运行  返回前面15的代码

    printf("Total training time: {:6.2f} sec".format(time_train), style='red')

    
    return ys_train,ys_test

if __name__ == '__main__':
    log_dir(args_global.train_config, args_global.data_prefix, git_branch, git_rev, timestamp)   ##untill 代码中的log_dir函数调用  用于控制台的调用
    train_params, train_phases, train_data, arch_gcn = parse_n_prepare(args_global)###将带有参数的文件yml  导入的函数中
    if 'eval_val_every' not in train_params:
        train_params['eval_val_every'] = EVAL_VAL_EVERY_EP  ##在globls.py代码中 EVAL_VAL_EVERY_EP = 1  
    model, minibatch, minibatch_eval, model_eval = prepare(train_data, train_params, arch_gcn)  ###调用前面的prepare的函数
    ys_train, ys_test = train(train_phases, model, minibatch, minibatch_eval, model_eval, train_params['eval_val_every'])

    
    

 