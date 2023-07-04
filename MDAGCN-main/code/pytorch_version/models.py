import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from graphsaint.utils import *
import graphsaint.pytorch_version.layers as layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GraphSAINT(nn.Module):
    def __init__(self, num_classes, arch_gcn, train_params, feat_full, label_full, cpu_eval=False):
        """
        Build the multi-layer GNN architecture.  构建多层GNN框架

        Inputs:
            num_classes         int, number of classes a node can belong to   一个节点可以属于的类的数目
            arch_gcn            dict, config for each GNN layer    配置每个GNN层
            train_params        dict, training hyperparameters (e.g., learning rate)训练参数
            feat_full           np array of shape N x f, where N is the total num of
                                nodes and f is the dimension for input node feature
                                形状为N x f的np数组，其中N为节点，f为输入节点特征的维数
            label_full          np array, for single-class classification, the shape
                                is N x 1 and for multi-class classification, the
                                shape is N x c (where c = num_classes)
                                用于单类分类，形状是nx1和多类分类，shape是N x c(其中c = num_classes)
            cpu_eval            bool, if True, will put the model on CPU. 如果为True，将把模型放在CPU上。

        Outputs:
            None
        """
        super(GraphSAINT,self).__init__()
        self.use_cuda = (args_global.gpu >= 0)
        if cpu_eval:
            self.use_cuda=False
        if "attention" in arch_gcn:
            if "gated_attention" in arch_gcn:
                if arch_gcn['gated_attention']:
                    self.aggregator_cls = layers.GatedAttentionAggregator   ##层聚合器
                    self.mulhead = int(arch_gcn['attention'])
            else:
                self.aggregator_cls = layers.AttentionAggregator
                self.mulhead = int(arch_gcn['attention'])
        else:
            self.aggregator_cls = layers.HighOrderAggregator
            self.mulhead = 1
        self.num_layers = len(arch_gcn['arch'].split('-'))
        self.weight_decay = train_params['weight_decay']   ###0
        self.dropout = train_params['dropout']        ####0.1
        self.lr = train_params['lr']                  ####0.001
        self.arch_gcn = arch_gcn
        self.sigmoid_loss = (arch_gcn['loss'] == 'sigmoid')
        self.feat_full = torch.from_numpy(feat_full.astype(np.float32))
        self.label_full = torch.from_numpy(label_full.astype(np.float32))
        if self.use_cuda:
            self.feat_full = self.feat_full.to(device)
            self.feat_full = self.feat_full.to(device)

        if not self.sigmoid_loss:
            self.label_full_cat = torch.from_numpy(label_full.argmax(axis=1).astype(np.int64))
            if self.use_cuda:
                self.label_full_cat = self.label_full_cat.to(device)

        self.num_classes = num_classes
        _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer \
                        = parse_layer_yml(arch_gcn, self.feat_full.shape[1])
        
        self.set_idx_conv()
        self.set_dims(_dims)

        self.loss = 0
        self.opt_op = None

        # build the model below  
        self.num_params = 0
        self.aggregators, num_param = self.get_aggregators()
        self.num_params += num_param
        self.conv_layers = nn.Sequential(*self.aggregators)
        self.classifier = layers.HighOrderAggregator(self.dims_feat[-1], self.num_classes,\
                            act='I', order=0, dropout=self.dropout, bias='bias')
        self.num_params += self.classifier.num_param
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def set_dims(self, dims):
        """
        Set the feature dimension / weight dimension for each GNN or MLP layer.
        We will use the dimensions set here to initialize PyTorch layers.
        设置每个GNN或MLP层的特征尺寸/权重尺寸。我们将使用这里设置的尺寸来初始化PyTorch层。

        Inputs:
            dims        list, length of node feature for each hidden layer

        Outputs:
            None
        """
        self.dims_feat = [dims[0]] + [
            ((self.aggr_layer[l]=='concat') * self.order_layer[l] + 1) * dims[l+1]
            for l in range(len(dims) - 1)
        ]
        self.dims_weight = [(self.dims_feat[l],dims[l+1]) for l in range(len(dims)-1)]

    def set_idx_conv(self):
        """
        Set the index of GNN layers for the full neural net. For example, if
        the full NN is having 1-0-1-0 arch (1-hop graph conv, followed by 0-hop
        MLP, ...). Then the layer indices will be 0, 2.
        为整个神经网络设置GNN层索引。例如,如果完整的NN有1-0-1-0拱门
        （1跳图convv，然后是0跳图延时,. .）
        然后层索引将是0,2
        """
        idx_conv = np.where(np.array(self.order_layer) >= 1)[0]
        idx_conv = list(idx_conv[1:] - 1)
        idx_conv.append(len(self.order_layer) - 1)
        _o_arr = np.array(self.order_layer)[idx_conv]
        if np.prod(np.ediff1d(_o_arr)) == 0:
            self.idx_conv = idx_conv
        else:
            self.idx_conv = list(np.where(np.array(self.order_layer) == 1)[0])


    def forward(self, node_subgraph, adj_subgraph):   ##向前传播
        feat_subg = self.feat_full[node_subgraph]
        label_subg = self.label_full[node_subgraph]
        label_subg_converted = label_subg if self.sigmoid_loss else self.label_full_cat[node_subgraph]
        _, emb_subg = self.conv_layers((adj_subgraph, feat_subg))
        emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)
        pred_subg = self.classifier((None, emb_subg_norm))[1]
        
        


        return pred_subg, label_subg, label_subg_converted    ####预测子图    标签子图  标签子图转换
        

    def _loss(self, preds, labels, norm_loss):
        """
        The predictor performs sigmoid (for multi-class) or softmax (for single-class)：
        预测执行sigmoid(用于多类)或softmax(用于单类)
        """
        if self.sigmoid_loss:
            norm_loss = norm_loss.unsqueeze(1)
            return torch.nn.BCEWithLogitsLoss(weight=norm_loss,reduction='sum')(preds, labels)
        else:
            _ls = torch.nn.CrossEntropyLoss(reduction='none')(preds, labels)
            return (norm_loss*_ls).sum()


    def get_aggregators(self):
        """
        Return a list of aggregator instances. to be used in self.build()
        返回聚合器实例列表。在self.build()中使用
        """
        num_param = 0
        aggregators = []
        for l in range(self.num_layers):
            aggr = self.aggregator_cls(
                    *self.dims_weight[l],
                    dropout=self.dropout,
                    act=self.act_layer[l],
                    order=self.order_layer[l],
                    aggr=self.aggr_layer[l],
                    bias=self.bias_layer[l],
                    mulhead=self.mulhead,
            )
            num_param += aggr.num_param
            aggregators.append(aggr)
        return aggregators, num_param

    ###预测
    def predict(self, preds):
        return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)   
   


    def train_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        """
        Forward and backward propagation
        前向传播和反向传播
        """
        self.train()
        self.optimizer.zero_grad()          
        preds, labels, labels_converted = self(node_subgraph, adj_subgraph)   
        loss = self._loss(preds, labels_converted, norm_loss_subgraph) 
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)  
        self.optimizer.step()

        return loss, self.predict(preds), labels

    def eval_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        """
        Forward propagation only
        只进行前向传播
        """
        self.eval()
        with torch.no_grad():     
            preds,labels,labels_converted = self(node_subgraph, adj_subgraph)   
            print('preds',preds.shape)
            loss = self._loss(preds,labels_converted,norm_loss_subgraph)
            
        return loss, self.predict(preds), labels
