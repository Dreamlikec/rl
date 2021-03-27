import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    actor神经网络，用于输出action的probability
    """

    def __init__(self, state_size, action_size, hidden=[128, 16]):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        log_prob = F.log_softmax(x, dim=1)
        return log_prob


class Critic(nn.Module):
    """
    Critic网络，用于输出单值来评价actor
    """

    def __init__(self, state_size, hidden=[128, 16]):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class Actor_Critic(nn.Module):
    """
    合并Actor和Critic网络共享前几层的权值，同时输出log_prob和value值
    """

    def __init__(self, state_size, action_size, hidden=[128, 16]):
        super(Actor_Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)
        self.fc4 = nn.Linear(hidden[1], 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        log_prob = F.log_softmax(self.fc3(x), dim=1)
        value = self.fc4(x)
        return log_prob, value
