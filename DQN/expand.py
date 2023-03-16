import random
from collections import deque

import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# define the hyperparameters for the DQN

EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 100000
TARGET_UPDATE = 1000
MEMORY_SIZE = 100000
LR = 0.0001

# define the replay buffer and the DQN
class ReplayBuffer:
    """
    The ReplayBuffer class is used to store transitions (state, action, reward, next_state, done) for training the DQN
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """"
    The DQN class is the neural network used to approximate the Q-values
    """
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the environment
env = gym.make('Pong-v4')

# Get the input shape and number of actions from the environment
input_shape = env.observation_space.shape
num_actions = env.action_space.n

# Initialize the DQN, target network, optimizer, and replay buffer
dqn = DQN(input_shape, num_actions)
target_dqn = DQN(input_shape, num_actions)
target_dqn.load_state_dict(dqn.state_dict())
optimizer = optim.Adam(dqn.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

# Define the epsilon-greedy exploration strategy
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.from_numpy(state).float().unsqueeze(0)
        q_values = dqn(state)
        return q_values.max(1)[1].item()



# Define the training loop
epsilon = EPSILON_START
episode_rewards = []
for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Select an action using epsilon-greedy exploration
        action = select_action(state, epsilon)

        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)  # the fourth value is not needed, so we use "_"

        # Update the total reward for the episode
        total_reward += reward

        # Add the transition to the replay buffer
        memory.push(state, action, reward, next_state, done)

        # Sample a batch of transitions from the replay buffer
        if len(memory) > BATCH_SIZE:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(BATCH_SIZE)

            # Convert the batches to PyTorch variables
            state_batch = torch.from_numpy(state_batch).float()
            action_batch = torch.from_numpy(action_batch).long()
            reward_batch = torch.from_numpy(reward_batch).float()
            next_state_batch = torch.from_numpy(next_state_batch).float()
            done_batch = torch.from_numpy(done_batch).float()

            # Compute the Q-values using the DQN
            q_values = dqn(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

            # Compute the target Q-values using the target DQN
            next_q_values = target_dqn(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values

            # Compute the loss and update the DQN
            loss = F.smooth_l1_loss(q_values, target_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the target DQN
            if episode % TARGET_UPDATE == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # Update the state
        state = next_state

    # Decay epsilon
    if epsilon > EPSILON_END:
        epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY

    # Print the total reward for the episode
    print('Episode %d: Total reward = %d' % (episode, total_reward))
    episode_rewards.append(total_reward)

    # Plot the episode rewards
    if episode % 10 == 0:
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()

