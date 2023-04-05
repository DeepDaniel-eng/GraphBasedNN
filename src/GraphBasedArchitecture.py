from torch import nn
from constants.ArchitectureConstants import *
from src.TransformerEncoder import TransformerEncoder
import torch

class GraphBasedArchitecture(nn.Module):
    
    def __init__(self, encoder_model, graph_architecture, model_dimension,
                 target_size, transformer_layers=4, num_heads = 4, dropout=0.2):
        """
        encoder_model: Torch model to encode input (preprocessor)
        graph: {
            node_id: {
                "node": NodeModel,
                "connections": {
                    node_id : edgeLayer_to_node_x
                }
            }
        }
        """
        super().__init__()
        self.encoder_model = encoder_model
        self.graph = graph_architecture
        self.to_evaluate_loss = nn.Linear(model_dimension, target_size)
        self.layernorm = nn.LayerNorm(3 * 32 * 32).cuda()
        self.transformer = TransformerEncoder(
            num_layers=transformer_layers,
            input_dim = model_dimension,
            dim_feedforward = 2* model_dimension,
            num_heads = num_heads,
            dropout = dropout
        )
        self.output_transf_net = nn.Sequential(
            nn.Linear(model_dimension, model_dimension),
            nn.LayerNorm(model_dimension),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(model_dimension, target_size)
        )
        self.out = nn.Sequential(
            nn.Linear(max(self.graph) * target_size, model_dimension),
            nn.LayerNorm(model_dimension),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(model_dimension, target_size)
        )
    
    def forward(self, input, batch_size=batch_size):
        # Generate encoding version
        graph_input = self.encoder_model(input)

        # Apply connection between encoder and all nodes and apply node to connected 
        base_id = 0
        for node in self.graph:
            node_info = self.graph[node]
            node = node_info["node"]
            connection = node_info["connections"][base_id]
            node(connection(graph_input, batch_size), base_id, batch_size)
        

        """
        Memory connection represents a status of the node after processing the base encoded input
        and the previously iterated nodes
        """

        # Apply all reltaionships between nodes
        for current_node in self.graph:
            to_be_processed = self.graph[current_node]["node"].memory_connection
            node_info = self.graph[current_node]
            node = self.graph[current_node]["node"]
            connections = node_info['connections']
            for connection, model in connections.items():
                if connection == 0:
                    continue
                after_connection = model(to_be_processed, batch_size)
                node_to_process = self.graph[connection]["node"]
                node_to_process(after_connection, current_node, batch_size)
        
        # Aggregate all connections
        # TODO: Learnable weights for each conection in the sum given the input
        tensor_to_apply_attention = torch.cat([self.graph[item]["node"].memory_connection.reshape(batch_size,1,-1) for item in self.graph ], axis=1)
        aggregated = self.transformer(tensor_to_apply_attention)
        out =  self.output_transf_net(aggregated).reshape(batch_size, -1)
        return self.out(out)

    
    def empty_connections(self):
        for node in self.graph:
            self.graph[node]["node"].memory_connection = None

