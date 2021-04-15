import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE


class Q_Network(nn.Module):
    def __init__(self, state_size, action_size, N, hidden=[64, 64]):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size * N)

        self.N = N
        self.action_size = action_size


    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.action_size, self.N)  # (batch_size,action_size,N)
        return x


class Visual_Q_Networks(nn.Module):
    """
    The input of this network should have shape (num_frame, 80, 80)
    """

    def __init__(self, num_frame, num_action, N):
        super(Visual_Q_Networks, self).__init__()
        self.conV1 = nn.Conv2d(in_channels=num_frame, out_channels=16, kernel_size=8, stride=4, padding=2)  # 16,20,20
        self.conV2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)  # 32,9,9
        self.fc1(32 * 81, 256)
        self.fc2(256, num_action * N)

        self.N = N
        self.action_size = num_action

    def forward(self, image):
        x = F.relu(self.conV1(image))
        x = F.relu(self.conV2(x))
        x = x.view(-1, 32 * 81)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # (batch_size,action_size * N)
        x = x.view(-1, self.action_size, self.N)  # (batch_size,action_size,N)
        return x
