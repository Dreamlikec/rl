import torch
import torch.optim as optim
import numpy as np
import math
from networks import *


class Agent_Discrete(object):
    def __init__(self, state_size, action_size, lr, beta, eps, lambda1, gamma, device, hidden=[256, 256],
                 share=False, mode="MC", use_critic=True, normalize=False):
        super(Agent_Discrete, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.lambda1 = lambda1
        self.device = device
        self.share = share
        self.mode = mode
        self.use_critic = use_critic
        self.normalize = normalize

        if self.share:
            self.actor_critic = Actor_Critic_discrete(self.state_size, self.action_size, hidden).to(self.device)
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr)
        else:
            self.actor = Actor_discrete(self.state_size, self.action_size, hidden).to(self.device)
            self.critic = Critic(self.state_size, hidden).to(self.device)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state).view(-1, self.state_size).to(self.device)
            if self.share:
                log_probs, values = self.actor_critic(state)
            else:
                log_probs = self.actor(state)
            probs = log_probs.exp().view(-1).cpu().numpy()
            action = np.random.choice(a=self.action_size, size=1, p=probs, replace=False)[0]
        return action

    def process_data(self, states, actions, rewards, dones, batch_size):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device).view(-1, 1)
        rewards = np.array(rewards)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device).view(-1, 1)

        N = states.size(0)
        steps = math.ceil(N / batch_size)
        log_probs = torch.zeros((N, self.action_size)).to(self.device)

        for step in range(steps):
            if self.share:
                output, _ = self.actor_critic(states[step * batch_size:(step + 1) * batch_size, :])
            else:
                output = self.actor(states[step * batch_size:(step + 1) * batch_size, :])
            log_probs[step * batch_size:(step + 1) * batch_size, :] = output

        log_probs = log_probs[:-1, :]

        return states, actions, log_probs.detach(), rewards, dones

    def learn(self, states, actions, log_probs, rewards, dones):
        if self.share:
            new_log_probs, states_values = self.actor_critic(states)
        else:
            new_log_probs = self.actor(states)
            states_values = self.critic(states)
        new_log_probs = new_log_probs[:-1, :]

        KL_Loss = log_probs.exp() * (log_probs - new_log_probs)
        KL_Loss = KL_Loss.sum(dim=1, keepdim=True)

        log_probs = torch.gather(log_probs, dim=1, index=actions)
        new_log_probs = torch.gather(new_log_probs, dim=1, index=actions)

        L = rewards.shape[0]
        with torch.no_grad():
            return_value = 0
            G = []
            if self.mode == "MC":
                for i in range(L - 1, -1, -1):
                    return_value = rewards[i] + self.gamma * (1 - dones[i]) * return_value
                    G.append(return_value)
                G = G[::-1]
                G = torch.tensor(G, dtype=torch.float).view(-1, 1).to(self.device)
            else:
                rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
                G = rewards + (1 - dones) * self.gamma * states_values[1:, :]

        Critic_Loss = 0.5 * (states_values[:-1, :] - G).pow(2).mean()

        with torch.no_grad():
            if self.use_critic:
                G = G - states_values[:-1, :]
            for i in range(L - 2, -1, -1):
                G[i] += G[i + 1] * self.lambda1 * self.gamma * (1 - dones[i])
            if self.normalize:
                G = (G - G.mean()) / (G.std() + 1e-10)

        ratio = (new_log_probs - log_probs).exp()
        Actor_Loss1 = ratio * G
        Actor_Loss2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * G
        Actor_Loss = -torch.min(Actor_Loss1, Actor_Loss2)
        Actor_Loss += self.beta * KL_Loss

        Actor_Loss = Actor_Loss.mean()

        if self.share:
            Loss = Actor_Loss + Critic_Loss
            self.optimizer.zero_grad()
            Loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1)
            self.optimizer.step()
        else:
            self.critic_optimizer.zero_grad()
            Critic_Loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            Actor_Loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.actor_optimizer.step()


class Agent_continuous(object):
    def __init__(self, state_size, action_size, lr, beta, eps, lambda1, gamma, device, hidden, share, mode, use_critic,
                 normalize):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.lambda1 = lambda1
        self.gamma = gamma
        self.device = device
        self.hidden = hidden
        self.share = share
        self.mode = mode
        self.use_critic = use_critic
        self.normalize = normalize
        if self.share:
            self.actor_critic = Actor_Critic_continuous(state_size, action_size, hidden).to(self.device)
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr)
        else:
            self.actor = Actor_continuous(state_size, action_size, hidden).to(self.device)
            self.critic = Critic(state_size, hidden).to(self.device)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr)

    def act(self, states):
        with torch.no_grad():
            states = torch.tensor(states).view(-1, self.state_size).to(self.device)
            if self.share:
                mu, log_std, _ = self.actor_critic(states)
            else:
                mu, log_std = self.actor(states)
            actions = torch.distributions.normal.Normal(mu, log_std.exp()).sample()
            actions = actions.cpu().numpy().reshape(-1)
        return actions

    def process_data(self, states, actions, rewards, dones, batch_size):
        actions.append(np.zeros(self.action_size))
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device).view(-1, self.action_size)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device).view(-1, 1)

        N = states.size(0)
        log_probs = torch.zeros((N, self.action_size)).to(self.device)
        old_mu = torch.zeros((N, self.action_size)).to(self.device)
        old_log_std = torch.zeros((N, self.action_size)).to(self.device)
        steps = math.ceil(N / batch_size)

        for step in range(steps):
            if self.share:
                mu, log_std, _ = self.actor_critic(states[step * batch_size:(step + 1) * batch_size, :])
            else:
                mu, log_std = self.actor(states[step * batch_size:(step + 1) * batch_size, :])
            distributions = torch.distributions.normal.Normal(mu, log_std.exp())
            log_probs[step * batch_size:(step + 1) * batch_size, :] = distributions.log_prob(
                actions[step * batch_size:(step + 1) * batch_size, :])
            old_mu[step * batch_size:(step + 1) * batch_size, :] = mu
            old_log_std[step * batch_size:(step + 1) * batch_size, :] = log_std

        log_probs = log_probs[:-1, :]
        actions = actions[:-1, :]
        old_mu = old_mu[:-1, :]
        old_log_std = old_log_std[:-1, :]
        log_probs = log_probs.sum(dim=1, keepdim=True)
        rewards = np.array(rewards)

        return states, actions, old_mu.detach(), old_log_std.detach(), log_probs.detach(), rewards, dones

    def learn(self, states, actions, old_mu, old_log_std, log_probs, rewards, dones):
        if self.share:
            new_mu, new_log_std, state_values = self.actor_critic(states)
            new_mu = new_mu[:-1, :]
            new_log_std = new_log_std[-1, :]
        else:
            new_mu, new_log_std = self.actor(states)
            state_values = self.critic(states)
            new_mu = new_mu[:-1, :]
            new_log_std = new_log_std[:-1, :]

        new_distribution = torch.distributions.normal.Normal(new_mu, new_log_std.exp())
        new_log_probs = new_distribution.log_prob(actions).sum(dim=1, keepdim=True)

        KL_Loss = new_log_std - old_log_std - 0.5 + (old_log_std.exp().pow(2) + (old_mu - new_mu).pow(2)) / (
                2 * new_log_std.exp().pow(2) + 1e1 - 6)
        KL_Loss = KL_Loss.sum(dim=1, keepdim=True)

        L = rewards.shape[0]

        with torch.no_grad():
            G = []
            return_value = 0
            if self.mode == "MC":
                for i in range(L - 1, -1, -1):
                    return_value = rewards[i] * self.gamma + return_value * (1 - dones[i])
                    G.append(return_value)
                G = G[::-1]
                G = torch.tensor(G, dtype=torch.float).to(self.device)
            else:
                rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
                G = rewards + state_values[1:, :] * self.gamma * (1 - dones)

        Critic_Loss = 0.5 * (G - state_values[:-1, :]).pow(2).mean()

        with torch.no_grad():
            if self.use_critic:
                G = G - state_values[:-1, :]
            for i in range(L - 2, -1, -1):
                G[i] += G[i + 1] * self.gamma * self.lambda1 * (1 - dones[i])
            if self.normalize:
                G = (G - G.mean()) / (G.std() + 1e-8)

        ratio = (new_log_probs - log_probs).exp()
        Actor_Loss1 = ratio * G
        Actor_Loss2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * G
        Actor_Loss = -torch.min(Actor_Loss1, Actor_Loss2)
        Actor_Loss += self.beta * KL_Loss

        Actor_Loss = Actor_Loss.mean()

        if self.share:
            Loss = Actor_Loss + Critic_Loss
            self.optimizer.zero_grad()
            Loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1)
            self.optimizer.step()
        else:
            self.critic_optimizer.zero_grad()
            Critic_Loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            Actor_Loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.actor_optimizer.step()
