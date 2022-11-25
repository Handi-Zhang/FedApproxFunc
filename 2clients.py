import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import os
from typing import Any, Dict, List
import copy
import random
import scipy.stats


class Client:
    def __init__(self,
                 client_id: Any,
                 model: torch.nn.Module,
                 loss: torch.nn.modules.loss._Loss,
                 optimizer: torch.optim.Optimizer,
                 optimizer_conf: Dict,
                 batch_size: int,
                 epochs: int,
                 server=None) -> None:
        self.client_id = client_id
        self.model = model
        self.loss = loss
        self.optimizer = optimizer(self.model.parameters(), **optimizer_conf)
        self.batch_size = batch_size
        self.epochs = epochs
        self.server = server
        self.accuracy = None
        self.total_loss = None

        self.data = None
        self.data_loader = None

    def setData(self, data):
        self.data = data
        self.data_loader = torch.utils.data.DataLoader(self.data,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)
        self.server.total_data += len(self.data)

    def update_weights(self):
        for eps in range(self.epochs):
            total_loss = 0
            total_batches = 0
            total_correct = 0

            for _, (feature, label) in enumerate(self.data_loader):
                feature = feature.to(device)
                label = label.to(device)
                
                y_pred = self.model(feature)
                
                loss = self.loss(y_pred, label)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
                

            self.total_loss = total_loss / total_batches
        

class Server:
    def __init__(self,
                 model: torch.nn.Module,
                 loss: torch.nn.modules.loss._Loss,
                 optimizer: torch.optim.Optimizer,
                 optimizer_conf: Dict,
                 n_client: int,
                 chosen_prob: float ,
                 local_batch_size: int,
                 local_epochs: int) -> None:

        # global model info
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_conf = optimizer_conf
        self.n_client = n_client
        self.local_batch_size = local_batch_size
        self.local_epochs = local_epochs
        self.total_data = 0

        # create clients
        self.client_pool: List[Client] = []
        self.create_client()
        self.chosen_prob = chosen_prob
        self.avg_loss = 0
        self.avg_acc = 0

    def create_client(self):
        # this function is reusable, so reset client pool is needed
        self.client_pool: List[Client] = []
        self.total_data = 0

        for i in range(self.n_client):
            model = copy.deepcopy(self.model)
            new_client = Client(client_id=i,
                                model=model,
                                loss=self.loss,
                                optimizer=self.optimizer,
                                optimizer_conf=self.optimizer_conf,
                                batch_size=self.local_batch_size,
                                epochs=self.local_epochs,
                                server=self)
            self.client_pool.append(new_client)

    def broadcast(self):
        model_state_dict = copy.deepcopy(self.model.state_dict())
        for client in self.client_pool:
            client.model.load_state_dict(model_state_dict)

    def aggregate(self):
        self.avg_loss = 0
        self.avg_acc = 0
        chosen_clients = random.sample(self.client_pool,
                                       int(len(self.client_pool) * self.chosen_prob))

        global_model_weights = copy.deepcopy(self.model.state_dict())
        for key in global_model_weights:
            global_model_weights[key] = torch.zeros_like(
                global_model_weights[key])

        for client in chosen_clients:
            client.update_weights()
            # print(f"Client {client.client_id}: Loss: {client.total_loss}")
            self.avg_loss += 1 / len(chosen_clients) * client.total_loss
            # self.avg_acc += 1 / len(chosen_clients) * client.accuracy
            local_model_weights = copy.deepcopy(client.model.state_dict())
            for key in global_model_weights:
                global_model_weights[key] += 1 / len(chosen_clients) * local_model_weights[key]

        self.model.load_state_dict(global_model_weights)
        

def split_a(a, c1, c2, n, func, clients):
    """
    a: sliding parameter to control level of non-iid;
    c1, c2: left and right bound of sampling interval;
    n: the number of points with in each sub-interval;
    func: function to sample training data from
    clinets: a list of Client class
    """

    split = sorted([c1 + (c2-c1) * a * 0.5 ** n for n in range(0, len(clients)-1)])

    # set the left most and right most interval
    tempx = np.linspace(c1, split[0], n)[:,None]
    tempy = func(tempx)
    tempdata = TensorDataset(torch.from_numpy(tempx).float(), torch.from_numpy(tempy).float())
    clients[0].setData(tempdata)

    tempx = np.linspace(split[-1], c2, n)[:,None]
    tempy = func(tempx)
    tempdata = TensorDataset(torch.from_numpy(tempx).float(), torch.from_numpy(tempy).float())
    clients[-1].setData(tempdata)

    for i in range(len(split)-1):
        tempx = np.linspace(split[i], split[i+1], n)[:,None]
        tempy = func(tempx)
        tempdata = TensorDataset(torch.from_numpy(tempx).float(), torch.from_numpy(tempy).float())
        clients[i+1].setData(tempdata)


# Define NN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,1)
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


def func(x):
    return -np.sin(10*np.pi*x)/(2*x+3) + (x+0.5)**4


ntest = 1000
x_test = torch.from_numpy(np.loadtxt('Gramacy&Lee(2012)_test.txt')[:, :1]).float()
y_test = torch.from_numpy(np.loadtxt('Gramacy&Lee(2012)_test.txt')[:, 1:]).float()
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=ntest, shuffle=False)

def test(federated_model):
    federated_model.eval()
    test_loss = 0
    for data, target in test_loader:
        output = federated_model(data)
        test_loss += F.mse_loss(output.view(-1), target, reduction='sum').item()
    test_loss /= len(test_loader.dataset)
    return test_loss

def run(num_clients, epochs, sliding_param, x_test):
    # initialize parameters
    # torch.manual_seed(1)
    chosen_prob = 1
    local_batch_size = 10
    criteria = nn.MSELoss()
    optimizer = optim.Adam
    optimizer_conf = dict(lr=0.001)

    n_client = num_clients
    local_epochs = 1
    epochs = epochs
    sliding_para = sliding_param

    # construct server and clients
    model = Net().to(device)

    server = Server(
        model=model,
        loss=criteria,
        optimizer=optimizer,
        n_client=n_client,
        chosen_prob=chosen_prob,
        optimizer_conf=optimizer_conf,
        local_batch_size=local_batch_size,
        local_epochs=local_epochs
        )

    # construct non-iid data
    clients = server.client_pool
    split_a(sliding_para, -1, 1, 100, func, clients)

    # train and save results
    for epoch in range(epochs):
        server.aggregate()
        server.broadcast()
        with torch.no_grad():
            testloss = test(server.model)

        if epoch % 100 == 0:
            print("Epoch: {}, Overall_loss: {}, test_loss: {}".format(epoch, server.avg_loss, testloss))

    y_pred = model(x_test).detach().numpy()
    return y_pred


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for a in [0.1, 0.2, 0.3, 0.4, 0.5]:
    pred = []
    for i in range(10):
        pred.append(run(num_clients=2, epochs=10000, sliding_param=a, x_test=x_test))
    np.savez( "2clients_{}.npz".format(a), pred=pred)