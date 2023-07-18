import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import HeteroGraphConv, GraphConv, WeightBasis


class RelGraphConvLayer(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = HeteroGraphConv({
            rel: GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names}, aggregate='sum')

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(th.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(th.split(weight, 1, dim=0))}
        else:
            wdict = {}
        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs
        hs = self.conv(g, inputs, mod_kwargs=wdict)    # dict

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class LocalNITCLayer(nn.Module):
    def __init__(self, i_dim, h_dim, o_dim):
        super(LocalNITCLayer, self).__init__()
        self.local_weight_m1 = nn.Linear(i_dim, h_dim)
        self.local_weight_m2 = nn.Linear(h_dim, h_dim)
        self.local_weight_m3 = nn.Linear(h_dim, o_dim)

        self.local_weight_d1 = nn.Linear(i_dim, h_dim)
        self.local_weight_d2 = nn.Linear(h_dim, h_dim)
        self.local_weight_d3 = nn.Linear(h_dim, o_dim)

    def forward(self, inputs_row, inputs_col):
        m_embedding1 = th.relu(self.local_weight_m1(inputs_row))
        m_embedding2 = th.relu(self.local_weight_m2(m_embedding1))
        m_embedding = th.relu(self.local_weight_m3(m_embedding2))

        d_embedding1 = th.relu(self.local_weight_d1(inputs_col))
        d_embedding2 = th.relu(self.local_weight_d2(d_embedding1))
        d_embedding = th.relu(self.local_weight_d3(d_embedding2))

        return m_embedding, d_embedding


class NMRDecoder(nn.Module):
    def __init__(self, num_relations, i_dim, h_dim=128, o_dim=64):
        super(NMRDecoder, self).__init__()
        self.num_relations = num_relations
        self.l_layers = nn.ModuleList()
        for _ in range(num_relations):
            self.l_layers.append(LocalNITCLayer(i_dim, h_dim, o_dim))

        self.global_weight_m1 = nn.Linear(o_dim, o_dim)
        self.global_weight_m2 = nn.Linear(o_dim, o_dim)

        self.global_weight_d1 = nn.Linear(o_dim, o_dim)
        self.global_weight_d2 = nn.Linear(o_dim, o_dim)

    def forward(self, inputs_row, inputs_col):
        outputs = []
        for k in range(self.num_relations):
            m_embedding, d_embedding = self.l_layers[k](inputs_row, inputs_col)

            m_embedding = th.relu(self.global_weight_m2(th.relu(self.global_weight_m1(m_embedding))))
            d_embedding = th.relu(self.global_weight_d2(th.relu(self.global_weight_d1(d_embedding))))

            outputs.append(m_embedding.mm(d_embedding.t()))
        outputs = th.cat(tuple(outputs))

        return outputs.reshape(self.num_relations, inputs_row.size(0), inputs_col.size(0))
