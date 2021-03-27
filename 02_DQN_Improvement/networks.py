import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_Network(nn.Module):
    def __init__(self, state_size, action_size, hidden=[64, 64], duel=False):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)
        self.duel = duel
        if self.duel:
            self.fc4 = nn.Linear(hidden[1], 1)

    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.duel:
            x1 = self.fc3(x)
            x1 = x1 - torch.max(x1, dim=1, keepdim=True)[0]  # set the max to be 0
            x2 = self.fc4(x)
            return x1 + x2
        else:
            x = self.fc3(x)
            return x


# class Visual_Q_Networks(nn.Module):
#     def __init__(self, num_frame, num_actions, duel=False):
#         super(Visual_Q_Networks, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=num_frame, out_channels=16, kernel_size=8, stride=4, padding=2)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
#         self.fc1 = nn.Linear(7 * 7 * 64, 512)
#         self.fc2 = nn.Linear(512, num_actions)
#         self.duel = duel
#         if self.duel:
#             self.fc3 = nn.Linear(512, 1)
#
#     def forward(self, image):
#         x = F.relu(self.conv1(image))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.view(-1, 7 * 7 * 64)
#         x = F.relu(self.fc1(x))
#         if self.duel:
#             x1 = self.fc2(x)
#             x1 = x1 - torch.max(x1, dim=1, keepdim=True)[0]
#             x2 = self.fc3(x)
#             return x1 + x2
#         else:
#             x = self.fc2(x)
#             return x

class Visual_Q_Networks(nn.Module):
    '''
    The input of this network should have shape (num_frame, 80, 80)
    '''

    def __init__(self, num_frame, num_action, duel=False):
        super(Visual_Q_Networks, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_frame, out_channels=16, kernel_size=8, stride=4, padding=2)  # 16, 20, 20
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)  # 32, 9, 9
        self.fc1 = nn.Linear(32 * 81, 256)
        self.fc2 = nn.Linear(256, num_action)
        self.duel = duel
        if self.duel:
            self.fc3 = nn.Linear(256, 1)

    def forward(self, image):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 81)
        x = F.relu(self.fc1(x))
        if self.duel:
            x1 = self.fc2(x)
            x1 = x1 - torch.max(x1, dim=1, keepdim=True)[0]
            x2 = self.fc3(x)
            return x1 + x2
        else:
            x = self.fc2(x)
            return x