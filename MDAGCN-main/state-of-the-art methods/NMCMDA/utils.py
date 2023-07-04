import random
import dgl
import torch as th
import numpy as np
import scipy.sparse as sp

from collections import defaultdict


def read_file(path):
    data = np.loadtxt(path, delimiter=',')
    return th.from_numpy(data).float()


def get_sim_index(mat):
    mx = sp.coo_matrix(mat)
    coords = np.vstack((mx.row, mx.col))
    return th.LongTensor(coords.tolist())


def get_tuple_index(mat):
    mx = sp.coo_matrix(mat)
    return zip(mx.row, mx.col)


def generate_hetero_graph(mm_index, dd_index, train_idx, mmda):
    sub_mmda = th.zeros(mmda.size())
    index = train_idx.t().long()
    sub_mmda[index[0], index[1], index[2]] = 1.

    row_list, col_list = [], []
    for ty in range(mmda.size(0)):
        md_matrix = sub_mmda[ty]
        row, col = np.where(md_matrix)
        row = th.from_numpy(row).long()
        col = th.from_numpy(col).long()
        row_list.append(row)
        col_list.append(col)

    data_dict = {('miRNA', 'msim', 'miRNA'): (mm_index[0], mm_index[1]),
                 ('disease', 'dsim', 'disease'): (dd_index[0], dd_index[1]),
                 ('miRNA', 'md_circulation', 'disease'): (row_list[0], col_list[0]),
                 ('miRNA', 'md_genetics', 'disease'): (row_list[1], col_list[1]),
                 ('miRNA', 'md_epigenetics', 'disease'): (row_list[2], col_list[2]),
                 ('miRNA', 'md_target', 'disease'): (row_list[3], col_list[3]),
                 ('miRNA', 'md_tissue', 'disease'): (row_list[4], col_list[4]),
                 ('miRNA', 'md_other', 'disease'): (row_list[5], col_list[5]),
                 ('disease', 'dm_circulation', 'miRNA'): (col_list[0], row_list[0]),
                 ('disease', 'dm_genetics', 'miRNA'): (col_list[1], row_list[1]),
                 ('disease', 'dm_epigenetics', 'miRNA'): (col_list[2], row_list[2]),
                 ('disease', 'dm_target', 'miRNA'): (col_list[3], row_list[3]),
                 ('disease', 'dm_tissue', 'miRNA'): (col_list[4], row_list[4]),
                 ('disease', 'dm_other', 'miRNA'): (col_list[5], row_list[5])}
    num_nodes_dict = {'miRNA': mmda.size(1), 'disease': mmda.size(2)}
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

    return g


def load_data(args):
    dataset = dict()

    circulation = read_file(args['data_path'] + '/circulation.csv')
    genetics = read_file(args['data_path'] + '/genetics.csv')
    epigenetics = read_file(args['data_path'] + '/epigenetics.csv')
    target = read_file(args['data_path'] + '/target.csv')
    tissue = read_file(args['data_path'] + '/tissue.csv')
    other = read_file(args['data_path'] + '/other.csv')

    dataset['mmda'] = th.tensor([circulation.tolist(), genetics.tolist(), epigenetics.tolist(), target.tolist(),
                                 tissue.tolist(), other.tolist()]).float()

    mm_matrix = read_file(args['data_path'] + '/mir_sim.csv')
    mm_edge_index = get_sim_index(mm_matrix)
    dataset['mm'] = {'data': mm_matrix, 'edge_index': mm_edge_index}

    dd_matrix = read_file(args['data_path'] + '/dis_sim.csv')
    dd_edge_index = get_sim_index(dd_matrix)
    dataset['dd'] = {'data': dd_matrix, 'edge_index': dd_edge_index}

    index_tmp = defaultdict(list)
    for k in range(len(dataset['mmda'])):
        neg_index = []
        for i in range(dataset['mmda'][k].size(0)):
            for j in range(dataset['mmda'][k].size(1)):
                if dataset['mmda'][k][i][j] == 0:
                    neg_index.append([k, i, j])
        random.shuffle(neg_index)
        neg_tensor = th.LongTensor(neg_index)
        neg_index = neg_tensor.split(int(np.ceil(neg_tensor.size(0) / args['fold'])), dim=0)

        for i in range(args['fold']):
            a = [j for j in range(args['fold'])]
            del a[i]
            train_neg_index = th.cat([neg_index[j] for j in a])

            index_tmp['index_{}'.format(k)].append({'train_neg': train_neg_index})

    dataset['index'] = []
    for i in range(args['fold']):
        dataset['index'].append({'train_neg': th.cat([index_tmp['index_{}'.format(k)][i]['train_neg']
                                                      for k in range(len(dataset['mmda']))])})

    dataset['type_index'] = []
    type_dataset = defaultdict(list)
    for k in range(dataset['mmda'].size(0)):
        for ind in get_tuple_index(dataset['mmda'][k]):
            type_dataset[ind].append(k)

    type_index = list(type_dataset.keys())
    random.shuffle(type_index)
    type_tensor = th.LongTensor(type_index)
    type_tensor = type_tensor.split(int(np.ceil(type_tensor.size(0) / args['fold'])), dim=0)

    for i in range(args['fold']):
        a = [j for j in range(args['fold'])]
        del a[i]
        test_index = type_tensor[i]
        train_index_temp = th.cat([type_tensor[j] for j in a])
        train_index = []
        for index in train_index_temp:
            for v in type_dataset[tuple(index.tolist())]:
                train_index.append([v, index.tolist()[0], index.tolist()[1]])

        test_target = np.zeros((len(test_index), len(dataset['mmda'])))
        test_index_temp = [[], []]
        for n, ind in enumerate(test_index):
            for v in type_dataset[tuple(ind.tolist())]:
                test_index_temp[0].append(n)
                test_index_temp[1].append(v)
        test_target[test_index_temp[0], test_index_temp[1]] = 1
        dataset['type_index'].append({'train': th.LongTensor(train_index), 'test': [test_index, test_target]})
    print('Data Preparation Have Finished！')

    return dataset
