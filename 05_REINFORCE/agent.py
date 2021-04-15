import torch
import torch.optim
import numpy as np
import math
from networks import *


class Agent(object):
    def __init__(self, state_size, action_size, lr, gamma, device, share=False, mode="MC", use_critic=False,
                 normalize=False):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.share = share
        self.mode = mode
        self.use_critic = use_critic
        self.normalize = normalize

        if self.share:  # 如果Actor和Critic共享网络
            self.Actor_critic = Actor_Critic(self.state_size, self.action_size).to(self.device)
            self.optimizer = torch.optim.Adam(self.Actor_critic.parameters(), lr)
        else:
            self.Actor = Actor(self.state_size, self.action_size).to(self.device)
            self.Critic = Critic(self.state_size).to(self.device)
            self.Actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr)
            self.Critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr)

    def act(self, state):
        """
        :param state: 输入当前状态
        :return: 根据policy网络也就是actor输出一个action给environment
        """
        with torch.no_grad():  # 这个过程中不需要求导
            state = torch.tensor(state, dtype=torch.float32).view(-1, self.state_size).to(self.device)
            if self.share:
                log_prob, _ = self.Actor_critic(state)
            else:
                log_prob = self.Actor(state)
            probabilities = log_prob.exp().view(-1).cpu().numpy()
            action = np.random.choice(a=self.action_size, size=1, replace=False, p=probabilities)[0]
        return action

    def process_data(self, states, actions, rewards, dones, batch_size):
        """
        :param states: 该transition下所有的state的集合，个数比其他集合多一个
        :param actions: 该transition下所有的action的集合
        :param rewards:  该transition 下所有的reward的集合
        :param dones: 是否结束transition
        :param batch_size: 批大小
        :return: 处理后的数据，后续带入损失函数中
        """
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device).view(-1, 1)

        N = states.size(0)
        log_probs = torch.zeros((N, self.action_size)).to(self.device)
        state_values = torch.zeros((N, 1)).to(self.device)

        step = math.ceil(N / batch_size)
        for i in range(step):
            if self.share:
                p, val = self.Actor_critic(states[i * batch_size: (i + 1) * batch_size, :])
            else:
                p = self.Actor(states[i * batch_size:(i + 1) * batch_size, :])
                val = self.Critic(states[i * batch_size:(i + 1) * batch_size, :])
            log_probs[i * batch_size: (i + 1) * batch_size, :] = p
            state_values[i * batch_size: (i + 1) * batch_size, :] = val
        log_probs = log_probs[:-1, :]
        log_probs = torch.gather(log_probs, dim=1, index=actions)

        L = len(rewards)
        rewards = np.array(rewards)
        discounts = self.gamma ** np.arange(L)
        discounted_reward = discounts * rewards

        return state_values, log_probs, rewards, discounted_reward, dones

    def learn(self, state_values, log_probs, rewards, dones):
        """
        这个的更新是一个transition结束后再更新
        更新参数，使用MSE更新Critic网络，使用A_t * log(A_t | S_t)更新Actor网络
        :param state_values: 一个transition中对应的所有状态的估计value
        :param log_probs: Actor输出的log_probabilities
        :param rewards: rewards的ndarray
        :param dones: transition是否结束
        :return: None
        """
        L = len(rewards)
        with torch.no_grad():
            G = []
            return_value = 0
            if self.mode == "MC":
                for i in range(L - 1, -1, -1):
                    return_value = rewards[i] + self.gamma * (1 - dones[i].detach().cpu().numpy()) * return_value
                    G.append(return_value)
                G = G[::-1]
                G = torch.tensor(G, dtype=torch.float).view(-1, 1).to(self.device)
            else:
                rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
                G = rewards + self.gamma * (1 - dones) * state_values[1:, :]
        Critic_Loss = 0.5 * (G - state_values[:-1,:]).pow(2).mean()

        with torch.no_grad():
            if self.use_critic:
                G = G - state_values[:-1, :]
            if self.normalize:
                G = (G - G.mean())/(G.std() + 0.0000001)

        Actor_loss = - log_probs * G
        Actor_loss = Actor_loss.mean()

        if self.share:
            Loss = Actor_loss + Critic_Loss
            self.optimizer.zero_grad()
            Loss.backward()
            self.optimizer.step()
        else:
            self.Critic_optimizer.zero_grad()
            Critic_Loss.backward()
            self.Critic_optimizer.step()
            self.Actor_optimizer.zero_grad()
            Actor_loss.backward()
            self.Actor_optimizer.step()