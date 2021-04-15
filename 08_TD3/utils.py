<<<<<<< HEAD
import random
import torch
import numpy as np
from collections import deque
from config import DEVICE

class ReplayBuffer(object):
    def __init__(self, max_len):
        self.memory = deque(maxlen=max_len)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None])).long().to(DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
=======
import random
import torch
import numpy as np
from collections import deque
from config import DEVICE

class ReplayBuffer(object):
    def __init__(self, max_len):
        self.memory = deque(maxlen=max_len)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None])).long().to(DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
>>>>>>> 8f0298de818621024d2322090959641c969b5f3a
