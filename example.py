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

LEARNING_RATE = 0.001
GAMMA = 0.99

actor = Actor(state_size, action_size)
critic = Critic(state_size)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)

MAX_EPISODES = 2000

for episode in range(MAX_EPISODES):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # --- 1. SELECTING ACTION ---
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = actor(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()

        # --- 2. INTERACTION WITH THE ENVIRONMENT ---
        next_state, reward, done, truncated, info = env.step(action.item())

        # --- 3. Calculating the Learning Signal ---
        current_state_value = critic(state_tensor)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        if done:
            next_state_value = torch.tensor([0.0])
        else:
            next_state_value = critic(next_state_tensor)

        advantage = reward + GAMMA * next_state_value - current_state_value

        # --- 4. NETWORK UPDATE (LEARNING MOMENT) ---
        # --- Update Critic ---
        critic_loss = advantage.pow(2)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # --- Update Actor ---
        log_prob = dist.log_prob(action)
        actor_loss = -log_prob * advantage.detach()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # --- PREPARING THE CYCLE FOR THE NEXT STEP---
        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()


