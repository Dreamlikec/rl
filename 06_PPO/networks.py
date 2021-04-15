import torch
import torch.nn as nn
import torch.nn.functional as f


class Actor_discrete(nn.Module):
    def __init__(self, state_size, action_size, hidden=[128]):
        super(Actor_discrete, self).__init__()
        hidden = [state_size] + hidden
        self.features = nn.ModuleList(nn.Linear(input, output) for input, output in zip(hidden[:-1], hidden[1:]))
        self.output = nn.Linear(hidden[-1], action_size)

    def forward(self, state):
        x = state
        for layer in self.features:
            x = f.relu(layer(x))
        log_probs = f.log_softmax(self.output(x), dim=1)
        return log_probs


class Actor_continuous(nn.Module):
    def __init__(self, state_size, action_size, hidden=[256, 256]):
        super(Actor_continuous, self).__init__()
        hidden = [state_size] + hidden
        self.features = nn.ModuleList(nn.Linear(input, output) for input, output in zip(hidden[:-1], hidden[1:]))
        self.mu = nn.Linear(hidden[-1], action_size)
        self.log_std = nn.Linear(hidden[-1], action_size)

    def forward(self, state):
        x = state
        for layer in self.features:
            x = f.relu(layer(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        return mu, log_std


class Critic(nn.Module):
    def __init__(self, state_size, hidden=[256, 256]):
        super(Critic, self).__init__()
        hidden = [state_size] + hidden
        self.features = nn.ModuleList(nn.Linear(input, output) for input, output in zip(hidden[:-1], hidden[1:]))
        self.output = nn.Linear(hidden[-1], 1)

    def forward(self, state):
        x = state
        for layer in self.features:
            x = f.relu(layer(x))
        values = self.output(x)
        return values


class Actor_Critic_discrete(nn.Module):
    def __init__(self, state_size, action_size, hidden=[128]):
        super(Actor_Critic_discrete, self).__init__()
        hidden = [state_size] + hidden
        self.features = nn.ModuleList(nn.Linear(input, output) for input, output in zip(hidden[:-1], hidden[1:]))
        self.actor = nn.Linear(hidden[-1], action_size)
        self.critic = nn.Linear(hidden[-1], 1)

    def forward(self, state):
        x = state
        for layer in self.features:
            x = f.relu(layer(x))
        log_prob = f.log_softmax(self.actor(x), dim = 1)
        values = self.critic(x)
        return log_prob, values


class Actor_Critic_continuous(nn.Module):
    def __init__(self, state_size, action_size, hidden=[256, 256]):
        super(Actor_Critic_continuous, self).__init__()
        hidden = [state_size] + hidden
        self.features = nn.ModuleList(nn.Linear(input, output) for input, output in zip(hidden[:-1], hidden[1:]))
        self.mu = nn.Linear(hidden[-1], action_size)
        self.log_std = nn.Linear(hidden[-1], action_size)
        self.critic = nn.Linear(hidden[-1], 1)

    def forward(self, state):
        x = state
        for layer in self.features:
            x = f.relu(layer(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        values = self.critic(x)
        return mu, log_std, values


