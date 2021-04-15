from networks import *
from replay_buffer import ProportionalReplayBuffer, ReplayBuffer, RankedReplayBuffer
from torch import optim
import numpy as np
import random


class Agent:
    def __init__(self, tau, gamma, batch_size, lr, state_size, actions_size, v_min, v_max, N, device, double=True,
                 visual=False, prioritized=False):
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.actions_size = actions_size
        self.v_min = v_min
        self.v_max = v_max
        self.N = N
        self.vals = torch.linspace(v_min, v_max, N).to(device)  # (batch_size ,N)
        self.unit = (v_max - v_min) / (N - 1)
        self.device = device
        self.double = double
        self.prioritized = prioritized

        if visual:
            self.Q_target = Visual_Q_Networks(state_size, actions_size, v_min, v_max, N).to(self.device)
            self.Q_local = Visual_Q_Networks(state_size, actions_size, v_min, v_max, N).to(self.device)
        else:
            self.Q_target = Q_Network(state_size, actions_size, v_min, v_max, N).to(self.device)
            self.Q_local = Q_Network(state_size, actions_size, v_min, v_max, N).to(self.device)

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
                _, actions_value = self.Q_local(state)
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

        local_log_prob, Q_local_value = self.Q_local(states)
        actions = actions.unsqueeze(1).repeat(1, 1, self.N)  # (batch_size,1,N)
        local_log_prob = torch.gather(input=local_log_prob, dim=1, index=actions)  # (batch_size,1,N)

        with torch.no_grad():
            target_log_prob, Q_target_value = self.Q_target(next_states)
            _, actions_target = torch.max(input=Q_target_value, dim=1, keepdim=True)  # (batch_size,1)
            actions_target = actions_target.unsqueeze(1).repeat(1, 1, self.N)  # (batch_size,1,1)
            target_log_prob = torch.gather(input=target_log_prob, dim=1, index=actions_target)  # (bath_size,1,N)
            target_log_prob = self.update_distribution(target_log_prob.exp(), rewards, self.gamma, dones)
            # (batch_size,1,N)

        loss = -local_log_prob * target_log_prob
        loss = loss.sum(dim=2, keepdim=False).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.prioritized:
            deltas = local_log_prob.sum(dim=2, keepdim=False) - target_log_prob.sum(dim=2, keepdim=False)
            deltas = np.abs(deltas.detach().cpu().numpy().reshape(-1))
            for i in range(self.batch_size):
                self.memory.insert(deltas[i], index_list[i])

    def update_distribution(self, prev_distribution, reward, gamma, dones):
        """
        :param prev_distribution: Q_target(X_t+1,a*)
        :param reward: 奖励
        :param gamma:  gamma
        :param dones: 是否结束
        :return:  更新话的分布
        """
        with torch.no_grad():
            reward = reward.view(-1, 1)  # (batch_size,1)
            batch_size = reward.size(0)
            assert prev_distribution.size(0) == batch_size
            new_vals = self.vals.view(1, -1) * gamma * (1 - dones) + reward  # (batch_size,N)
            new_vals = torch.clamp(new_vals, self.v_min, self.v_max).to(self.device)
            lower_indexes = torch.floor((new_vals - self.v_min) / self.unit).long().to(self.device)  # (batch_size,N)
            upper_indexes = torch.min(lower_indexes + 1, other=torch.tensor(self.N - 1).to(self.device)).to(
                self.device)  # (batch_size,N)
            lower_vals = self.vals[lower_indexes].to(self.device)  # (batch_size,N)
            lower_distances = 1 - torch.min((new_vals - lower_vals) / self.unit,
                                            other=torch.tensor(1, dtype=torch.float32).to(self.device)).to(
                self.device)  # (batch_size,N)
            transition = torch.zeros((batch_size, self.N, self.N)).to(self.device)
            first_dim = torch.tensor(range(batch_size), dtype=torch.long).view(-1, 1).repeat(1, self.N).view(-1).to(
                self.device)  # (bath_size * N)

            second_dim = torch.tensor(range(self.N), dtype=torch.long).repeat(batch_size).view(-1).to(
                self.device)  # (batch_size * N)
            transition[first_dim, second_dim, lower_indexes.view(-1)] += lower_distances.view(-1)
            transition[first_dim, second_dim, upper_indexes.view(-1)] += 1 - lower_distances.view(-1)
            if len(prev_distribution.size()) == 2:
                prev_distribution = prev_distribution.unsqueeze(1)  # (batch_size,action_size,N)
            return torch.bmm(prev_distribution, transition)  # (batch_size,action_size,N)

    def soft_update(self):
        for Q_target_param, Q_local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            Q_target_param.data.copy_(self.tau * Q_local_param.data + (1.0 - self.tau) * Q_target_param.data)
