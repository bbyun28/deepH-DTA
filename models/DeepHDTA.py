import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
from graph import HGANLayer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class Dense_Block(nn.Module):
    def __init__(self, in_channels,n_filter,k, dropRate=0.0):
        super(Dense_Block, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.bn = nn.BatchNorm2d(num_channels = in_channels)
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = n_filter, kernel_size = k)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = n_filter, kernel_size = k)
        self.conv3 = nn.Conv1d(in_channels = 64, out_channels = n_filter, kernel_size = k)
        
    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv1 = F.dropout(out, p=self.droprate, training=self.training)
        conv2 = self.relu(self.conv2(conv1))
        conv2 = F.dropout(out, p=self.droprate, training=self.training)
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        conv3 = F.dropout(out, p=self.droprate, training=self.training)
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
        out= SELayer(c3_dense)
        return out

class Transition_Layer(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(Transition_Layer, self).__init__() 
        self.relu = nn.ReLU(inplace = True) 
        self.bn = nn.BatchNorm2d(num_features = out_channels) 
        self.conv = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False) 
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0) 
    def forward(self, x):
        bn = self.bn(self.relu(self.conv(x))) 
        out = self.avg_pool(bn) 
    return out 
# GAT  model
class DeepHDTA(torch.nn.Module):
    def __init__(self,num_meta_paths,num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(DeepHDTA, self).__init__()

        # graph layers
        self.layers = nn.ModuleList()
        self.layers.append(HGANLayer(num_meta_paths, num_features_xd, num_features_xd, 12, dropout))
        self.layers.append(HGANLayer(num_meta_paths, num_features_xd * 12, output_dim, 10, dropout))
        self.layers.append(HGANLayer(num_meta_paths, num_features_xd * 10, output_dim, 8, dropout))
        self.layers.append(HGANLayer(num_meta_paths, num_features_xd * 8, output_dim, 6, dropout))
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        
        # 1D ConvLSTM on SMILES sequences
        self.W_rnn = convolutional_rnn.Conv1dLSTM(in_channels=100,  # Corresponds to input size
                                   out_channels=100,  # Corresponds to hidden size
                                   kernel_size=3, num_layers=1,bidirectional=True,
                                   dropout=0.4,
                                   batch_first=True)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.denseblock1 = Dense_Block(in_channels=1000, out_channels=n_filters,6, dropRate=0.2) 
        self.denseblock2 = Dense_Block(in_channels=500, out_channels=n_filters,9, dropRate=0.2) 
        self.denseblock3 = Dense_Block(in_channels=500, out_channels=n_filters,12, dropRate=0.2) 
        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels = 196, out_channels = n_filters) 
        self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels = 196, out_channels = n_filters) 
        self.fc_xt1 = nn.Linear(32*121, output_dim)
        # combined layers
        self.fc1 = nn.Linear(1024, 768)
        self.fc2 = nn.Linear(768, 512)
        self.out = nn.Linear(512, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

     def _make_transition_layer(self, layer, in_channels, out_channels): 
            modules = [] 
            modules.append(layer(in_channels, out_channels)) 
            return nn.Sequential(*modules)
     def rnn(self, xs):
        xs = torch.unsqueeze(xs, 0)
        xs, h = self.W_rnn(xs)
        xs = torch.relu(xs)
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)
    
    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

          
        return self.predict(h)
        x = F.dropout(x, p=0.2, training=self.training)
        for gnn in self.layers:
            x = gnn(x, edge_index)
        x = F.dropout(x, p=0.2, training=self.training)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)
        
        """smile vector with convLSTM."""
        smiles=data.smiles
        smile_vectors = self.embed_smile(smiles)
        after_smile_vectors = self.rnn(smile_vectors)

        # protein input feed-forward:
        target = data.target
        embedded_xt = self.embedding_xt(target)
        denseout = self.denseblock1(embedded_xt) 
        denseout = self.transitionLayer1(denseout) 
        denseout = self.denseblock2(denseout) 
        denseout = self.transitionLayer2(denseout) 
        denseout = self.denseblock3(denseout) 
        denseout = self.transitionLayer3(denseout)

        # flatten
        xt = denseout.view(-1, 32 * 121)
        xt = self.fc_xt1(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
