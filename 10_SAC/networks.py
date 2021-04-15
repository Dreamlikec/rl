import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Q_Network(nn.Module):
    def __init__(self, state_size, action_size, hidden=[256, 256]):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def hidden_unit(layer):
    fan_in = layer.weight.data.size()[0]
    bound = 1 / math.sqrt(fan_in)
    lim = (-bound, bound)
    return lim


class Actor(nn.Module):
    def __init__(self, state_size, action_size, std=1, hidden=[256, 256]):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)
        self.fc4 = nn.Linear(hidden[1], action_size)
        self.std = std
        self.reset_parameters()

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        means = self.fc3(x)
        log_std = torch.tanh(self.fc4(x))
        return means, self.std * log_std

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_unit(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_unit(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_unit(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_unit(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        self.fc4.bias.data.uniform_(-3e-3, 3e-3)


class Critic(nn.Module):
    def __init__(self, state_size, hidden=[256, 256]):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)
        self.reset_parameters()

    def forward(self, states):
        x = states
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_unit(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_unit(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_unit(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_unit(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
