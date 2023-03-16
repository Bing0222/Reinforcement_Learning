# @Time    : 2023/3/16
# @Author  : Bing

import gym
# env = gym.make("LunarLander-v2", render_mode="human")
# env.action_space.seed(42)
#
# observation, info = env.reset(seed=42)
#
# for _ in range(1000):
#     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
#
#     if terminated or truncated:
#         observation, info = env.reset()
#
# env.close()

env = gym.make("CartPole-v1", render_mode="human")

# Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)
n_states = env.observation_space.shape[0]  # 4
# Discrete(2)
n_actions = env.action_space.n  # 2

