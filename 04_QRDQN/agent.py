from networks import *
from replay_buffer import ProportionalReplayBuffer, ReplayBuffer, RankedReplayBuffer
from torch import optim
import numpy as np
import random


class Agent:
    def __init__(self, tau, gamma, batch_size, lr, state_size, actions_size, kappa, N, device, double=True,
                 visual=False, prioritized=False):
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.actions_size = actions_size
        self.N = N
        self.tau_q = torch.linspace(0, 1, N + 1)
        self.tau_q = (self.tau_q[1:] + self.tau_q[:-1]) / 2
        self.tau_q = self.tau_q.to(device).unsqueeze(0)  # (1,N)
        self.kappa = kappa
        self.device = device
        self.double = double
        self.prioritized = prioritized

        if visual:
            self.Q_target = Visual_Q_Networks(state_size, actions_size, N).to(self.device)
            self.Q_local = Visual_Q_Networks(state_size, actions_size, N).to(self.device)
        else:
            self.Q_target = Q_Network(state_size, actions_size, N).to(self.device)
            self.Q_local = Q_Network(state_size, actions_size, N).to(self.device)

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
                actions_value = self.Q_local(state)  # (batch_size,action_size,N)
                actions_value = actions_value.sum(dim=2,keepdims=False).view(-1)
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
        actions = actions.to(self.device)  # (batch_size,1)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        quantiles_local = self.Q_local(states)
        actions = actions.unsqueeze(1).repeat(1, 1, self.N)  # (batch_size,1,N)
        quantiles_local = torch.gather(input=quantiles_local, dim=1, index=actions)  # (batch_size,1,N)

        with torch.no_grad():
            quantiles_target = self.Q_target(next_states)
            _, actions_target = torch.max(input=quantiles_target.sum(dim=2, keepdims=True), dim=1,
                                          keepdim=True)  # (batch_size,1,1)
            actions_target = actions_target.repeat(1, 1, self.N)  # (batch_size,1,1)
            quantiles_target = torch.gather(input=quantiles_target, dim=1, index=actions_target)  # (batch_size,1,N)
            quantiles_target = rewards.unsqueeze(1).repeat(1, 1, self.N) + self.gamma * \
                               (1 - (dones.unsqueeze(1).repeat(1, 1, self.N))) * quantiles_target  # (batch_size,1,N)

        diff = quantiles_target.permute(0, 2, 1) - quantiles_local  # (batch_size,N,N)
        loss = self.huber_loss(diff, self.kappa, self.tau_q)  # (batch_size,N,N)
        loss = loss.mean(dim=2, keepdim=False).sum(dim=1, keepdim=False).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.prioritized:
            deltas = quantiles_target.sum(dim=2, keepdim=False) - quantiles_local.sum(dim=2, keepdim=False)
            deltas = np.abs(deltas.detach().cpu().numpy().reshape(-1))
            for i in range(self.batch_size):
                self.memory.insert(deltas[i], index_list[i])

    def huber_loss(self, u, kappa, tau):
        if kappa > 0:
            flag = (u.abs() < kappa).float()
            huber = 0.5 * u.pow(2) * flag + kappa * (u.abs() - 0.5 * kappa) * (1 - flag)
        else:
            huber = u.abs()
        loss = (tau - (u < 0).float()).abs() * huber
        return loss

    def soft_update(self):
        for Q_target_param, Q_local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            Q_target_param.data.copy_(self.tau * Q_local_param.data + (1.0 - self.tau) * Q_target_param.data)
