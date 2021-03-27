import numpy as np
import random
from config import *
from utils import SumTree
import torch
import bisect


class ReplayBuffer(object):
    def __init__(self, capacity, batch_size=64):
        self.capacity = capacity
        self.memory = [None for _ in range(capacity)]
        self.ind_max = 0

    def remember(self, transition):
        ind = self.ind_max % self.capacity
        self.memory[ind] = transition
        self.ind_max += 1

    def sample(self, k):
        '''
        return sampled transitions. Make sure that there are at least k transitions stored before calling this method
        '''
        index_set = random.sample(list(range(len(self))), k)
        states = torch.from_numpy(np.vstack([self.memory[ind][0] for ind in index_set])).float()
        actions = torch.from_numpy(np.vstack([self.memory[ind][1] for ind in index_set])).long()
        rewards = torch.from_numpy(np.vstack([self.memory[ind][2] for ind in index_set])).float()
        next_states = torch.from_numpy(np.vstack([self.memory[ind][3] for ind in index_set])).float()
        dones = torch.from_numpy(np.vstack([self.memory[ind][4] for ind in index_set]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return min(self.ind_max, self.capacity)


class ProportionalReplayBuffer(object):
    def __init__(self, capacity, batch_size=64):
        """
        初始化一个Proportional的ReplayBuffer类，需要存储需要的transition，记录各个transition的TD-error,定义Alpha,
        :param capacity: ReplayBuffer的容量大小
        :param batch_size: 从ReplayBuffer中抽样出来的batch_size的大小
        """
        self.alpha = ALPHA
        self.epsilon = EPSILON
        self.capacity = capacity
        self.memory = [None for _ in range(capacity)]
        self.tree = SumTree(self.capacity)
        self.max_index = 0
        self.default_delta = TD_INIT
        self.batch_size = batch_size

    def remember(self, transition):
        """
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        index = self.max_index % self.capacity
        self.memory[index] = transition
        # delta = max(self.tree.nodeVal[-self.capacity:])
        # if delta == 0:
        #     delta = self.default_delta
        delta = self.default_delta + EPSILON - self.tree.nodeVal[index + self.capacity - 1]
        self.tree.update(delta, index)
        self.max_index += 1

    def sample(self, batch_size):
        """
        根据batch_size的大小均匀采样到各个区间，但区间的长度不一样，很显然，区间长度更大的叶子结点在SumTree中更容易被所索引到
        :return: 返回所有采样到的内容
        """
        index_set = [self.tree.retrieve(self.tree.nodeVal[0] * random.random()) for _ in range(batch_size)]
        probs = torch.from_numpy(
            np.vstack([self.tree.nodeVal[ind + self.capacity - 1] / self.tree.nodeVal[0] for ind in index_set])).float()
        states = torch.from_numpy(np.vstack([self.memory[ind][0] for ind in index_set])).float()
        actions = torch.from_numpy(np.vstack([self.memory[ind][1] for ind in index_set])).long()
        rewards = torch.from_numpy(np.vstack([self.memory[ind][2] for ind in index_set])).float()
        next_states = torch.from_numpy(np.vstack([self.memory[ind][3] for ind in index_set])).float()
        dones = torch.from_numpy(np.vstack([self.memory[ind][4] for ind in index_set]).astype(np.uint8)).float()

        return index_set, states, actions, rewards, next_states, dones, probs

    def insert(self, delta, index):
        change = (delta + self.epsilon) ** self.alpha - self.tree.nodeVal[index + self.capacity - 1]
        self.tree.update(change, index)

    def __len__(self):
        return min(self.capacity, self.max_index)


class RankedReplayBuffer(object):
    def __init__(self, capacity, batch_size=64):
        """
        初始化一个RankedReplayBuffer,需要顶一个TD-error的有序数组,定义一个segments用于划分区间,total_error用于累加所有error,
        cumulative_errors用于将各个各个阶段的累加error存储起来并用于划分segments
        :param capacity: ReplayBuffer的容量大小
        :param batch_size: 从ReplayBuffer中抽样出来的batch_size的大小
        """
        self.alpha = ALPHA
        self.epsilon = EPSILON
        self.capacity = capacity
        self.memory = [None for _ in range(self.capacity)]
        self.max_index = 0
        self.default_delta = 1.
        self.batch_size = batch_size
        self.total_error = 0.
        self.cumulative_weights = []
        self.errors = []
        self.memory_to_rank = [None for _ in range(self.capacity)]
        self.segments = [-1] + [None for _ in range(self.batch_size)]

    def remember(self, transition):
        """
         更新ReplayBuffer，原则是轮番剔除插入，将新的transition插入进来，原则是将对应位置的TD-error替换成当前erros有序数组中最大
         的那个error，然后将error重新排序，同时保存好排序后每个error对应的index
        :param transition: transition包含state, action, reward, next_state, done, 需要存储到响应的ReplayBuffer的Memory中
        """
        index = self.max_index % self.capacity
        if self.max_index < self.capacity:  # 当memory中的transition没存满的时候
            self.total_error = (1 / (self.max_index + 1)) ** self.alpha
            self.cumulative_weights.append(self.total_error)
            self.update_segments()
        else:
            self.pop(index)

        self.memory[index] = transition
        max_error = -self.errors[0][0]
        self.insert(max_error, index)
        self.max_index += 1

    def pop(self, index):
        idx = self.memory_to_rank[index]
        self.errors.pop(idx)
        self.memory_to_rank[index] = None
        for i in range(idx, len(self.errors)):
            self.memory_to_rank[self.errors[idx][1]] -= 1

    def insert(self, error, index):
        sort_idx = bisect.bisect_left(self.errors, (-error, index))
        self.memory_to_rank[index] = sort_idx
        self.errors.insert(sort_idx, (-error, index))
        for i in range(sort_idx + 1, len(self.errors)):
            self.memory_to_rank[self.errors[sort_idx][1]] += 1

    def update_segments(self):
        if self.max_index + 1 < self.batch_size:
            return None
        for i in range(self.batch_size):
            sort_index = bisect.bisect_left(self.cumulative_weights, self.total_error * ((i + 1) / self.batch_size))
            self.segments[i] = max(sort_index, self.segments[i] + 1)

    def sample(self):
        index_list = [random.randint(self.segments[i] + 1, self.segments[i + 1]) for i in range(self.batch_size)]
        probs = torch.from_numpy(
            np.vstack([(1 / 1 + index) ** self.alpha / self.total_error for index in index_list])).float()
        index_list = [self.errors[index][1] for index in index_list]

        states = torch.from_numpy(np.vstack([self.memory[index][0] for index in index_list])).float()
        actions = torch.from_numpy(np.vstack([self.memory[index][1] for index in index_list])).long()
        rewards = torch.from_numpy(np.vstack([self.memory[index][2] for index in index_list])).float()
        next_states = torch.from_numpy(np.vstack([self.memory[index][3] for index in index_list])).float()
        dones = torch.from_numpy(np.vstack([self.memory[index][4] for index in index_list])).float()

        return index_list, probs, states, actions, rewards, next_states, dones
