from torch import nn
from constants.ArchitectureConstants import *

class GraphEdge(nn.Module):
    
    def __init__(self, layer, relu=False):
        super().__init__()
        self.layer = layer
        self.relu = nn.ReLU() if relu else None
    
    def forward(self, x, batch_size=batch_size):
        out = self.layer(x, batch_size)
        if self.relu is not None:
            out = self.relu(out)
        return out
