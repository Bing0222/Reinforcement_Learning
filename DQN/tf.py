# @Time    : 2023/3/16
# @Author  : Bing


import tensorflow as tf
import numpy as np
import gym

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory_size = 100000

# Environment parameters
env_name = 'CartPole-v1'
env = gym.make(env_name)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Replay memory
memory = []

# Build the neural network
class DQNAgent:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(24, input_dim=state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='linear')
        ])
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    def act(self, state):
        if np.random.rand() <= epsilon:
            return env.action_space.sample()
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        memory.append((state, action, reward, next_state, done))
        if len(memory) > memory_size:
            del memory[0]

    def replay(self):
        if len(memory) < batch_size:
            return
        minibatch = np.random.choice(memory, batch_size, replace=False)
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        targets = self.model.predict(states)
        q_values_next = self.model.predict(next_states)
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + discount_factor * np.amax(q_values_next[i])
        self.model.fit(states, targets, epochs=1, verbose=0)

    def decay_epsilon(self):
        global epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

# Initialize agent and training loop
agent = DQNAgent()
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    time_step = 0
    while not done:
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        time_step += 1
        agent.replay()
        agent.decay_epsilon()
    print("Episode: {}, Time steps: {}, Epsilon: {:.4f}".format(episode, time_step, epsilon))

env.close()
