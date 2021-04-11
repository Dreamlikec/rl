import torch
import torch.optim as optim
import torch.nn.functional as f
import numpy as np
import random
from collections import deque
from config import *
from network import Q_Network


class Agent:
    def __init__(self, state_size, action_size, alpha, bs, lr, tau, gamma, device):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.bs = bs
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.Q_local = Q_Network(state_size, action_size).to(self.device)
        self.Q_target = Q_Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.Q_local.parameters(), lr)
        self.soft_update(1)
        self.memory = deque(maxlen=100000)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float).view(1, -1).to(self.device)
            Q_values = self.Q_local(state)
            action_value = f.log_softmax(Q_values / self.alpha, dim=1)
            action_distribution = torch.distributions.Categorical(action_value.exp())
            action = action_distribution.sample().cpu().item()
            return action

    def state_value(self, Q_values):
        with torch.no_grad():
            Q_values = Q_values / self.alpha
            Q_max, _ = Q_values.max(dim=1, keepdims=True)
            Q_values = Q_values - Q_max
            state_values = Q_max + Q_values.exp().sum(dim=1, keepdims=True).log()
            state_values = self.alpha * state_values
            return state_values

    def learn(self):
        experiences = random.sample(self.memory, self.bs)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences])).long().to(self.device)

        Q_values = self.Q_local(states)
        Q_values = torch.gather(Q_values, dim=-1, index=actions)

        with torch.no_grad():
            next_Q_values = self.Q_target(next_states)
            next_values = self.state_value(next_Q_values)
            Q_targets = rewards + self.gamma * (1 - dones) * next_values

        loss = (Q_values - Q_targets).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.tau)

    def soft_update(self, tau):
        for local_param, target_param in zip(self.Q_local.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(tau * local_param + (1 - tau) * target_param)