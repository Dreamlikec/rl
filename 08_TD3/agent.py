import torch
import torch.optim as optim
import numpy as np
from utils import *
from networks import *


class Agent(object):
    def __init__(self, env, lr, device, tau=0.001, delay=2, step=1, batch_size=64, exploration_level=0.2,
                 smoothing_level=0.1, smoothing_max=0.5, batch_norm=True, clip=True, initialize=True,
                 hidden=[256, 256]):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        self.env = env
        self.lr = lr
        self.device = device
        self.tau = tau
        self.delay = delay
        self.step = step
        self.batch_size = batch_size
        self.exploration_level = exploration_level
        self.smoothing_level = smoothing_level
        self.smoothing_max = smoothing_max
        self.clip = clip
        self.gamma = 0.99
        self.memory = ReplayBuffer(max_len=int(1e6))
        self.count = 0

        self.actor_local = Actor(self.state_size, self.action_size, batch_norm, initialize, hidden).to(DEVICE)
        self.actor_target = Actor(self.state_size, self.action_size, batch_norm, initialize, hidden).to(DEVICE)
        self.critic1_local = Critic(self.state_size, self.action_size, batch_norm, initialize, hidden).to(DEVICE)
        self.critic2_local = Critic(self.state_size, self.action_size, batch_norm, initialize, hidden).to(DEVICE)
        self.critic1_target = Critic(self.state_size, self.action_size, batch_norm, initialize, hidden).to(DEVICE)
        self.critic2_target = Critic(self.state_size, self.action_size, batch_norm, initialize, hidden).to(DEVICE)

        self.soft_update(self.critic1_local, self.critic1_target, 1)
        self.soft_update(self.critic2_local, self.critic2_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr)
        self.critic1_optimizer = optim.Adam(self.critic1_local.parameters(), lr)
        self.critic2_optimizer = optim.Adam(self.critic2_local.parameters(), lr)

    @staticmethod
    def soft_update(local_network: nn.Module, target_network: nn.Module, tau):
        for local_layer, target_layer in zip(local_network.modules(), target_network.modules()):
            for local_param, target_param in zip(local_layer.parameters(), target_layer.parameters()):
                target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

            try:
                target_layer.running_mean = tau * local_layer.running_mean + (1 - tau) * target_layer.running_mean
                target_layer.running_var = tau * local_layer.running_var + (1 - tau) * target_layer.running_var

            except:
                pass

    def act(self, state, i):
        state = torch.tensor(state, dtype=torch.float).to(self.device).view(1, -1)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).detach().view(-1).cpu().numpy()
            noise = self.exploration_noise().numpy()
        self.actor_local.train()
        action += noise / math.sqrt(i)
        action = np.clip(action, -1, 1)
        return action

    def exploration_noise(self):
        mean = torch.zeros(self.action_size)
        std = self.exploration_level * torch.ones(self.action_size)
        return torch.normal(mean, std)

    def smooting_noise(self):
        with torch.no_grad():
            mean = torch.zeros(self.action_size).to(self.device)
            std = self.smoothing_level * torch.ones(self.action_size)
            noise = torch.normal(mean, std)
            noise = torch.clamp(noise, -self.smoothing_max, self.smoothing_max)
        return noise

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 先把target计算出来
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            noise = self.smooting_noise()
            next_actions += noise
            next_actions = torch.clamp(next_actions, -1, 1)
            next_values1 = self.critic1_target(next_states, next_actions)
            next_values2 = self.critic2_target(next_states, next_actions)
            next_values = torch.min(next_values1, next_values2)
            target_values = rewards + self.gamma * (1 - dones) * next_values

        local1_values = self.critic1_local(states, actions)
        local2_values = self.critic2_local(states, actions)
        Loss = 0.5 * ((local1_values - target_values).pow(2).mean() + (local2_values - target_values).pow(2).mean())
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        Loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.critic1_local.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.critic2_local.parameters(), 2)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        self.count = (self.count + 1) % self.delay
        if self.count == 0:
            Loss = -self.critic1_local(states, self.actor_local(states)).mean()
            self.actor_optimizer.zero_grad()
            Loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
            self.actor_optimizer.step()

            self.soft_update(self.actor_local, self.actor_target, self.tau)
            self.soft_update(self.critic1_local, self.critic1_target, self.tau)
            self.soft_update(self.critic2_local, self.critic2_target, self.tau)

