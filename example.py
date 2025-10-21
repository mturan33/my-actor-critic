import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

env = gym.make("CartPole-v1", render_mode='human')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

state, info = env.reset()
print("Initial Status:", state)
random_action = env.action_space.sample()
next_state, reward, done, _, _ = env.step(random_action)
print("Random Post-Action Status:", next_state)

env.close()


