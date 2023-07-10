from layers import *


class RGCNNMRModel(nn.Module):
    def __init__(self, feat_dim, num_nodes, rel_names, num_bases, num_hidden_layers=4):
        super(RGCNNMRModel, self).__init__()
        self.feat_dim = feat_dim
        self.num_nodes = num_nodes
        self.rel_names = rel_names
        self.num_relations = int(len(rel_names) / 2 - 1)
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        self.encoder_layers = nn.ModuleList()
        for i in range(self.num_hidden_layers):
            self.encoder_layers.append(RelGraphConvLayer(
                self.feat_dim, self.feat_dim, self.rel_names, self.num_bases, activation=F.relu))

        self.decoder_layers = NMRDecoder(self.num_relations, feat_dim)

    def forward(self, args, g, h=None, blocks=None):
        if h is None:
            th.manual_seed(1)
            f_m = th.randn(self.num_nodes[0], self.feat_dim).to(args['device'])
            f_d = th.randn(self.num_nodes[1], self.feat_dim).to(args['device'])
            h = {'miRNA': f_m, 'disease': f_d}
        if blocks is None:
            for layer in self.encoder_layers:
                h = layer(g.to(args['device']), h)
        else:
            for layer, block in zip(self.encoder_layers, blocks):
                h = layer(block, h)

        output = self.decoder_layers(h['miRNA'], h['disease'])

        return output
