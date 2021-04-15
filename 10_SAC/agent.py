import torch
import torch.optim as optim
import numpy as np
import math
from torch.distributions import Normal
from utils import ReplayBuffer
from networks import *


class Agent(object):
    def __init__(self, env, device, N=1, lr=0.001, alpha=1, std=1, update_frequency=4, tau=0.001, gamma=0.99,
                 buffer_size=int(1e6),
                 batch_size=128, log_in_V_loss=True, log_in_pi_loss=True):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.gamma = gamma
        self.env = env
        self.device = device
        self.N = N
        self.lr = lr
        self.alpha = alpha
        self.std = std
        self.update_frequency = update_frequency
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.log_in_V_loss = log_in_V_loss
        self.log_in_pi_loss = log_in_pi_loss

        self.Q1 = Q_Network(self.state_size, self.action_size).to(device)
        self.Q2 = Q_Network(self.state_size, self.action_size).to(device)
        self.pi = Actor(self.state_size, self.action_size, std).to(device)
        self.V_local = Critic(self.state_size).to(device)
        self.V_target = Critic(self.state_size).to(device)
        self.soft_update(self.V_local, self.V_target, 1)

        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr)
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr)
        self.V_optimizer = optim.Adam(self.V_local.parameters(), lr)

        self.memory = ReplayBuffer(buffer_size)

    def soft_update(self, local_model, target_model, tau):
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            local_param.data.copy_(tau * target_param.data + (1 - tau) * local_param.data)

    def act(self, states):
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float).view(1, -1).to(self.device)
            means, log_stds = self.pi(states)
            stds = log_stds.exp()
            actions = Normal(means, stds).sample()
            actions = torch.tanh(actions)
            actions = actions.cpu().numpy().reshape(-1)
        return actions

    def reparameters(self, means, stds):
        distribution = Normal(means, stds)
        actions = distribution.rsample()
        log_probs = distribution.log_prob(torch.sum(actions.detach(), dim=1, keepdim=True))
        news_actions = torch.tanh(actions)
        return news_actions, log_probs

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states.to(self.device)
        actions.to(self.device)
        rewards.to(self.device)
        next_states.to(self.device)
        dones.to(self.device)

        # update Q of theta
        with torch.no_grad():
            next_states_values = self.V_target(states)
            expected_values = rewards + (1 - dones) * self.gamma * next_states_values
        Q1_values = self.Q1(states, actions)
        Q1_loss = 0.5 * (Q1_values - expected_values).pow(2).mean()
        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.Q1_optimizer.step()

        Q2_values = self.Q2(states, actions)
        Q2_loss = 0.5 * (Q2_values - expected_values).pow(2).mean()
        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.Q2_optimizer.step()

        if self.N > 1:
            states = states.repeat(1, self.N).view(self.batch_size * self.N, self.state_size)
        means, log_stds = self.pi(states)
        stds = log_stds.exp().to(self.device)
        new_actions, log_probs = self.reparameters(means, stds)
        Q1_values = self.Q1(states, new_actions)
        Q2_values = self.Q2(states, new_actions)
        Q_values = torch.min(Q1_values, Q2_values)

        # update V of psai
        V_values = self.V_local(states)
        if self.log_in_V_loss:
            V_loss = 0.5 * (V_values - Q_values.detach() + self.alpha * log_probs.detach()).pow(2).mean()
        else:
            V_loss = 0.5 * (V_values - Q_values.detach).pow(2).mean()

        self.V_optimizer.zero_grad()
        V_loss.backward()
        self.V_optimizer.step()
        self.soft_update(self.V_local, self.V_target, self.tau)

        # update pi of fai
        if self.log_in_pi_loss:
            pi_loss = (self.alpha * log_probs - Q_values).mean()
        else:
            # this means that maximize Q a kind likes DDPG
            pi_loss = -Q_values.mean()

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()
