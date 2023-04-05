from torch import nn
from constants.ArchitectureConstants import *

class GraphNode(nn.Module):

    def __init__(self, identifier, layer, post_process_func=None):
        super().__init__()
        self.layer = layer
        self.post_process_func = post_process_func
        self.memory_connection = None
        self.identifier = identifier
        self.aggregate = aggregate
        self.layernorm = nn.LayerNorm(3 * 32 * 32).cuda()
        self.aggregation = nn.Sequential(
            nn.Linear()
        )

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
            self.memory_connection = self.layernorm(self.memory_connection.reshape(batch_size, -1) + hidden.reshape(batch_size, -1)).reshape(batch_size, 3 , 32, 32)
        