from torch import nn
from constants.ArchitectureConstants import *

class GraphNode(nn.Module):

    def __init__(self, identifier, layer, post_process_func=None, aggregate='sum'):
        super().__init__()
        self.layer = layer
        self.post_process_func = post_process_func
        self.memory_connection = None
        self.identifier = identifier
        self.aggregate = aggregate
        self.batchnorm = nn.BatchNorm2d(3).cuda()
        self.node_prediction = nn.Linear(32*32, 10).cuda()
        self.node_activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3).cuda()

    def forward(self, x, node_from_id, batch_size=batch_size):
        hidden = self.layer(x, batch_size)
        if self.post_process_func is not None:
            hidden = self.post_process_func(hidden)
        if self.memory_connection is None:
            self.memory_connection = hidden
        else:
            # Temporally aggregating by sum because of computing limitation
            # Alternative is concatenation
            # TODO: Learnable weights depending on input
            self.memory_connection = self.batchnorm((self.memory_connection + hidden)/2)
    
    def transform_lattent_space(self):
        return self.node_activation(self.dropout((self.node_prediction(self.memory_connection.sum(axis=1).reshape(batch_size, -1)))))