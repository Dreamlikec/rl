from collections import deque
from config import DEVICE
from copy import copy
import numpy as np
import random
import torch


class ReplayBuffer(object):
    def __init__(self, max_len):
        self.memory = deque(maxlen=max_len)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences])).long().to(DEVICE)
        return states, actions, rewards, next_states, dones


class OUNnoise:
    def __init__(self, action_size, mu=0, theta=0.15, sigma=0.5):
        self.action_size = action_size
        self.mu = mu * np.ones(action_size)
        self.sigma = sigma
        self.theta = theta
        self.reset()

    def reset(self):
        self.state = copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.uniform(size=self.action_size)
        self.state = x + dx
        return self.state
