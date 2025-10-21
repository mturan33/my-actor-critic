import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.network(state)

env = gym.make("CartPole-v1", render_mode='human')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

state, info = env.reset()
print("Initial Status:", state)
random_action = env.action_space.sample()
next_state, reward, done, _, _ = env.step(random_action)
print("Random Post-Action Status:", next_state)

env.close()


