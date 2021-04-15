<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import math


def hidden_init(layer):
    dimension = layer.weight.data.size()[0]
    lim = 1 / math.sqrt(dimension)
    bound = (-lim, lim)
    return bound


class Critic(nn.Module):
    def __init__(self, state_size, action_size, batch_norm, initialize, hidden=[256, 256]):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.batch_norm = batch_norm
        if initialize:
            self.initialize()

    def forward(self, state, action):
        state = torch.cat([state, action], dim=1)
        if self.batch_norm:
            x = f.relu(self.bn1(self.fc1(state)))
        else:
            x = f.relu(self.fc1(state))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def initialize(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)
        self.fc3.bias.data.uniform_(-3e-4, 3e-4)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, batch_norm, initialize, hidden=[256, 256]):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.bn2 = nn.BatchNorm1d(hidden[1])
        self.batch_norm = batch_norm
        if initialize:
            self.initialize()

    def forward(self, state):
        x = state
        if self.batch_norm:
            x = f.relu(self.bn1(self.fc1(x)))
            x = f.relu(self.bn2(self.fc2(x)))
        else:
            x = f.relu(self.fc1(x))
            x = f.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def initialize(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
=======
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import math


def hidden_init(layer):
    dimension = layer.weight.data.size()[0]
    lim = 1 / math.sqrt(dimension)
    bound = (-lim, lim)
    return bound


class Critic(nn.Module):
    def __init__(self, state_size, action_size, batch_norm, initialize, hidden=[256, 256]):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.batch_norm = batch_norm
        if initialize:
            self.initialize()

    def forward(self, state, action):
        state = torch.cat([state, action], dim=1)
        if self.batch_norm:
            x = f.relu(self.bn1(self.fc1(state)))
        else:
            x = f.relu(self.fc1(state))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def initialize(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)
        self.fc3.bias.data.uniform_(-3e-4, 3e-4)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, batch_norm, initialize, hidden=[256, 256]):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.bn2 = nn.BatchNorm1d(hidden[1])
        self.batch_norm = batch_norm
        if initialize:
            self.initialize()

    def forward(self, state):
        x = state
        if self.batch_norm:
            x = f.relu(self.bn1(self.fc1(x)))
            x = f.relu(self.bn2(self.fc2(x)))
        else:
            x = f.relu(self.fc1(x))
            x = f.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def initialize(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
>>>>>>> 8f0298de818621024d2322090959641c969b5f3a
