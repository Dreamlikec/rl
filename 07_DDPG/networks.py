import torch
import math
import torch.nn as nn
import torch.nn.functional as f


def hidden_uniform(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1 / math.sqrt(fan_in)
    bound = (-lim, lim)
    return bound


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
        x = self.fc3(x)
        action = torch.tanh(x)
        return action  # used for giving a action between -1 and 1

    def initialize(self):
        self.fc1.weight.data.uniform_(*hidden_uniform(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_uniform(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_uniform(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_uniform(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, batch_norm, initialize, hidden=[256, 256]):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0] + action_size, hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.batch_norm = batch_norm
        if initialize:
            self.initialize()

    def forward(self, state, action):
        x = state
        if self.batch_norm:
            x = f.relu(self.bn1(self.fc1(x)))
        else:
            x = f.relu(self.fc1(x))
        x = torch.cat([x, action], 1)
        x = f.relu(self.fc2(x))
        value = self.fc3(x)
        return value  # used for critic to an action

    def initialize(self):
        self.fc1.weight.data.uniform_(*hidden_uniform(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_uniform(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_uniform(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_uniform(self.fc2))
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)
        self.fc3.bias.data.uniform_(-3e-4, 3e-4)
