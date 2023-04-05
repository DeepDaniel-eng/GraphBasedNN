from tqdm import tqdm
import wandb
import torch
from torch import nn
import sys
sys.path[0] = sys.path[0].replace('train','')
from constants.ArchitectureConstants import *
from utils.generate_base_graph import generate_architecture_1, get_optimizer_from_graph, get_scheduler
from data.set_up_data import set_up_CIFAR10_data
from constants.ArchitectureConstants import *

if wandb_:
    user = "daniel"
    project = "GraphBasedArquitecture"
    display_name = "experiment-2023-03-25 2"
    wandb.init(project=project, name=display_name)


def test_accuracy(net, testloader, bs, wandb_ = wandb_):
    correct = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        net.eval()
        for images, labels in testloader:
            images, labels = images.to("cuda"), labels.to("cuda")

            # calculate outputs by running images through the network
            outputs = net(images, bs)

            # the class with the highest energy is what we choose as prediction
            predicted = torch.max(outputs.data, 1)[1]
            curr_correct = (predicted == labels).sum().item()
            correct += curr_correct
            if wandb_:
                wandb.log({"TestCorrectBatch": correct/64 * 100}, commit=False)
            net.empty_connections()

        
        if wandb_:
            wandb.log({"TesAccuracy": correct / len(testloader.dataset)})

    return correct / len(testloader.dataset)
    


def train(net, trainLoader, testloader, criterion, optimizer, scheduler, epochs, bs):
    for epoch in range(epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        correct = 0
        total_loss_per_epoch = 0
        for i, data in tqdm(enumerate(trainLoader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs, bs)
            loss = criterion(outputs, labels)
            predicted = torch.max(outputs.data, 1)[1]
            current_correct = (predicted == labels).sum().item()
            correct += current_correct    
            wandb.log({"BatchCorrect": current_correct/64 * 100}, commit=False)
            
            for node in net.graph:
                predicted = net.graph[node]["node"].transform_lattent_space()
                loss += criterion(predicted, labels)/ len(net.graph)
            loss.backward()

            optimizer.step()
            net.empty_connections()
            # print statistics
            running_loss += loss.item()
            if i % 200 == 0 and i!= 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                total_loss_per_epoch += running_loss
                if wandb_:
                    to_commit = {"RunningLoss": running_loss / 200 }
                    wandb.log(to_commit)
                running_loss = 0.0
        scheduler.step()
        if wandb_:
            wandb.log({"TrainAccuracy": correct/len(trainLoader.dataset), "Epoch": epoch, "TotalLossPerEPoch": total_loss_per_epoch})
        test_accuracy(net, testloader, batch_size, True)



    print('Finished Training')

if __name__ == '__main__':
    graph_arch = generate_architecture_1()
    trainloader, testloader = set_up_CIFAR10_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer_from_graph(graph_arch)
    scheduler = get_scheduler(optimizer)
    train(graph_arch, trainloader, testloader, criterion, optimizer,scheduler, n_iters, batch_size)