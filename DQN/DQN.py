# @Time    : 2023/3/15
# @Author  : Bing

import collections
import random

import numpy as np
import torch.nn as nn
import torch.optim
import torch.nn.functional as F


class ReplayBuffer():
    def __init__(self, capacity: int):
        # creat a list(first into first out), max len: capacity
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        transition = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transition)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class Net(nn.Module):
    def __init__(self, n_states:int, n_hidden:int, n_action:int):
        super(Net, self).__init__()
        # [b,n_states] -> [b,n_hidden]
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_action)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class DQN:
    def __init__(self, n_states:int, n_hidden:int, n_actions:int,
                 lr:float, gamma, epsilon, target_update, device):
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma #discount factor
        self.epsilon = epsilon
        self.target_update = target_update # frequency of update
        self.device = device
        self.count = 0

        # build two model(same sturtc different paramers)
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

    def take_action(self, state):
        # 维度扩充，给行增加一个维度，并转换为张量shape=[1,4]
        state = torch.Tensor(state[np.newaxis, :])
        # 如果小于该值就取最大的值对应的索引
        if np.random.random() < self.epsilon:  # 0-1
            # 前向传播获取该状态对应的动作的reward
            actions_value = self.q_net(state)
            # 获取reward最大值对应的动作索引
            action = actions_value.argmax().item()  # int
        # 如果大于该值就随机探索
        else:
            # 随机选择一个动作
            action = np.random.randint(self.n_actions)
        return action

    def update(self, transition_dict): # get batch
        # array_shape=[batch,4]
        states = torch.tensor(transition_dict['state'], dtype=torch.float)
        # [b,1]
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        # [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_state'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # 奖励 + 折扣因子 * 下个时刻的最大回报
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
