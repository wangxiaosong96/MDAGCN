import torch as th
import warnings
import argparse

from torch import nn, optim
from sklearn.metrics import precision_score, recall_score
from utils import load_data, generate_hetero_graph
from model import RGCNNMRModel

warnings.filterwarnings("ignore")


class TransDataset(object):
    def __init__(self, dataset):
        self.data_set = dataset

    def __getitem__(self, n):
        return (self.data_set['type_index'][n]['train'], self.data_set['index'][n]['train_neg'],
                self.data_set['type_index'][n]['test'][0], self.data_set['type_index'][n]['test'][1])


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, true_index, neg_index, target, preds):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(preds, target)

        return (1 - args['alpha']) * loss_sum[true_index.t().tolist()].sum() + \
                args['alpha'] * loss_sum[neg_index.t().tolist()].sum()


def train(gs, model, train_data, optimizer, sizes):
    model.train()
    cost = MyLoss()
    true_index = train_data[0].to(args['device'])
    neg_index = train_data[1].to(args['device'])

    def train_epoch():
        model.zero_grad()
        predict = model(args, gs)
        loss = cost(true_index, neg_index, sizes['target'].to(args['device']), predict)
        loss.backward()
        optimizer.step()
        return loss

    for epoch in range(1, args['epoch']+1):
        train_reg_loss = train_epoch()


def main():
    data_set = load_data(args)
    split_data = TransDataset(data_set)
    sizes = {'num_m': data_set['mm']['data'].size(0),
             'num_d': data_set['dd']['data'].size(0),
             'num_types': len(data_set['mmda']),
             'target': data_set['mmda'],
             'mm': data_set['mm']['data'],
             'dd': data_set['dd']['data']}
    for i in range(args['fold']):
        print('-' * 50)
        print('Training_%d' % (i+1))
        gs = generate_hetero_graph(data_set['mm']['edge_index'],
                                   data_set['dd']['edge_index'],
                                   split_data[i][0],
                                   data_set['mmda'])
        all_relations = ['msim', 'dm_circulation', 'dm_genetics', 'dm_epigenetics', 'dm_target', 'dm_tissue',
                         'dm_other',
                         'dsim', 'md_circulation', 'md_genetics', 'md_epigenetics', 'md_target', 'md_tissue',
                         'md_other']
        model = RGCNNMRModel(args['in_size'],
                             [sizes['num_m'], sizes['num_d']],
                             all_relations,
                             num_bases=-1).to(args['device'])
        optimizer = optim.Adam(model.parameters(), lr=0.0009)
        train(gs, model, split_data[i], optimizer, sizes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/MCD6')
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--in_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--out_size', type=int, default=64)
    args = parser.parse_args().__dict__

    args['device'] = 'cuda' if th.cuda.is_available() else 'cpu'

    main()
