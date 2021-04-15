from networks import *
from replay_buffer import ProportionalReplayBuffer, ReplayBuffer, RankedReplayBuffer
from torch import optim
import numpy as np
import random


class Agent:
    def __init__(self, tau, gamma, batch_size, lr, state_size, actions_size, device, double=True, duel=False,
                 visual=False, prioritized=False):
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.actions_size = actions_size
        self.device = device
        self.double = double
        self.duel = duel
        self.prioritized = prioritized

        if visual:
            self.Q_target = Visual_Q_Networks(state_size, actions_size, duel=duel).to(self.device)
            self.Q_local = Visual_Q_Networks(state_size, actions_size, duel).to(self.device)
        else:
            self.Q_target = Q_Network(state_size, actions_size, duel=duel).to(self.device)
            self.Q_local = Q_Network(state_size, actions_size, duel=duel).to(self.device)

        self.optimizer = optim.Adam(self.Q_local.parameters(), lr=lr)
        self.soft_update()

        if self.prioritized:
            self.memory = ProportionalReplayBuffer(int(1e5), batch_size)
            # self.memory = RankedReplayBuffer(int(1e5), batch_size)
        else:
            self.memory = ReplayBuffer(int(1e5), batch_size)

    def act(self, state, epsilon=0.1):
        """
        :param state: 输入的input
        :param epsilon:  以epsilon的概率进行exploration, 以 1 - epsilon的概率进行exploitation
        :return:
        """
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                actions_value = self.Q_local(state)
            return np.argmax(actions_value.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.actions_size))

    def learn(self):
        if self.prioritized:
            index_list, states, actions, rewards, next_states, dones, probs = self.memory.sample(self.batch_size)
            w = 1 / len(self.memory) / probs
            w = w / torch.max(w)
            w = w.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            w = torch.ones(actions.size())
            w = w.to(self.device)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        Q_local_values = self.Q_local(states)
        Q_local_values = torch.gather(Q_local_values, dim=-1, index=actions)

        with torch.no_grad():
            Q_targets_values = self.Q_target(next_states)
            if self.double:
                max_actions = torch.max(input=self.Q_local(next_states), dim=1, keepdim=True)[1]
                Q_targets_values = torch.gather(Q_targets_values, dim=1, index=max_actions)
            else:
                Q_targets_values = torch.max(input=Q_targets_values, dim=1, keepdim=True)[0]

            Q_targets_values = rewards + self.gamma * (1 - dones) * Q_targets_values

        deltas = Q_local_values - Q_targets_values

        loss = (w * deltas).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.prioritized:
            deltas = np.abs(deltas.detach().cpu().numpy().reshape(-1))
            for i in range(self.batch_size):
                self.memory.insert(deltas[i], index_list[i])

    def soft_update(self):
        for Q_target_param, Q_local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            Q_target_param.data.copy_(self.tau * Q_local_param.data + (1.0 - self.tau) * Q_target_param.data)
