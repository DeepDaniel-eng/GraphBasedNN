from src.GraphNode import GraphNode
from src.GraphEdge import GraphEdge
from src.ArchitectureLayers import ConvBlock, ConnectionBlock
from src.GraphBasedArchitecture import GraphBasedArchitecture
from torch import nn
import torch.optim as optim
from constants.ArchitectureConstants import *

def generate_graph_architecture(n_nodes, base_model_node=nn.Linear,base_model_edege=nn.Linear, base_params_node=(1024,1024), base_params_edge=(1024 * 15, 1024 * 3)):
    return {
        node_id: {
                    "node": GraphNode(node_id, ConvBlock().to("cuda")),
                    "connections": {
                        node_id : GraphEdge(ConnectionBlock(), True).to("cuda")
                     for node_id in range( n_nodes + 1)}
                }
             for node_id in range(1, n_nodes + 1)}


def generate_architecture_1():

    graph = generate_graph_architecture(graph_size)
    encoder_model = nn.Sequential(ConnectionBlock(), ConvBlock())
    graph_arch = GraphBasedArchitecture(encoder_model, graph, hidden_dimension_size, output_size).to("cuda")
    params = [x for x in graph_arch.parameters()] + \
            [graph_arch.graph[x]['node'].parameters() for x in graph_arch.graph] + \
            [t for x in graph_arch.graph for y in graph_arch.graph[x]['connections'] for t in graph_arch.graph[x]['connections'][y].parameters()]
    params_non_generator = [x for x in params if type(x) == nn.parameter.Parameter]
    params_generator = [x for x in params if type(x) != nn.parameter.Parameter]
    params_to_append_generator = [t for x in params_generator for t in x]
    optimizer = optim.Adam(params_non_generator + params_to_append_generator, lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

def get_optimizer_from_graph(graph_arch):
    params = [x for x in graph_arch.parameters()] + \
            [graph_arch.graph[x]['node'].parameters() for x in graph_arch.graph] + \
            [t for x in graph_arch.graph for y in graph_arch.graph[x]['connections'] for t in graph_arch.graph[x]['connections'][y].parameters()]
    params_non_generator = [x for x in params if type(x) == nn.parameter.Parameter]
    params_generator = [x for x in params if type(x) != nn.parameter.Parameter]
    params_to_append_generator = [t for x in params_generator for t in x]
    optimizer = optim.Adam(params_non_generator + params_to_append_generator, lr=lr)
    return optimizer

def get_scheduler(optimizer):
    return optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, scheduler_gamma=0.2)

if __name__ == '__main__':
    input_size = 512
    hidden_dimension_size = 1024
    output_size = 10
    graph = generate_graph_architecture(5)
    base_model = nn.Conv2d(3, 15, (3,3) ,padding=1)
    encoder_model = nn.Sequential(ConnectionBlock(), ConvBlock())
    graph_arch = GraphBasedArchitecture(encoder_model, graph, hidden_dimension_size, output_size).to("cuda")

