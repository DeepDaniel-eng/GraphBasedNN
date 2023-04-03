from torch import nn
from constants.ArchitectureConstants import *

class GraphBasedArchitecture(nn.Module):
    
    def __init__(self, encoder_model, graph_architecture, model_dimension, target_size):
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
        At this point each node has processed the encoding output so its memory looks like
        memory = {0 : <encoded_plus_node_tensor>}
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
        aggregated = self.layernorm(sum(self.graph[item]["node"].memory_connection for item in self.graph).reshape(batch_size, -1))
        return self.to_evaluate_loss(aggregated)
    
    def empty_connections(self):
        for node in self.graph:
            self.graph[node]["node"].memory_connection = None

