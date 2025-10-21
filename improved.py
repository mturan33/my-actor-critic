import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

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

# --- VISUALIZATION SETUP ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Actor-Critical Training Process")

# Chart 1: Awards
ax1.set_ylabel("Total Rewards")
ax1.set_title("Total Rewards per Episode")
line_rewards, = ax1.plot([], [])

# Graph 2: Losses
ax2.set_xlabel("Episode")
ax2.set_ylabel("Loss")
ax2.set_title("Actor and Critical Losses")
line_actor_loss, = ax2.plot([], [], label="Actor Loss", color="blue")
line_critic_loss, = ax2.plot([], [], label="Critical Loss", color="red")
ax2.legend()

episode_numbers = deque(maxlen=100)
all_rewards = deque(maxlen=100)
all_actor_losses = deque(maxlen=100)
all_critic_losses = deque(maxlen=100)

LEARNING_RATE = 0.001
GAMMA = 0.99

env = gym.make("CartPole-v1", render_mode='human')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

actor = Actor(state_size, action_size)
critic = Critic(state_size)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)

MAX_EPISODES = 2000

for episode in range(MAX_EPISODES):
    state, info = env.reset()
    done = False
    total_reward = 0

    episode_reward = 0
    episode_actor_losses = []
    episode_critic_losses = []

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

        # --- COLLECTING DATA ---
        episode_reward += reward
        episode_actor_losses.append(actor_loss.item())
        episode_critic_losses.append(critic_loss.item())

        # --- PREPARING THE CYCLE FOR THE NEXT STEP---
        state = next_state
        total_reward += reward

    # --- VISUALIZATION ---
    # 1. Add data to main lists
    episode_numbers.append(episode)
    all_rewards.append(episode_reward)
    # Take the average of losses in this episode and add it to the list
    all_actor_losses.append(np.mean(episode_actor_losses))
    all_critic_losses.append(np.mean(episode_critic_losses))

    # 2. Update line data
    line_rewards.set_data(list(episode_numbers), list(all_rewards))
    line_actor_loss.set_data(list(episode_numbers), list(all_actor_losses))
    line_critic_loss.set_data(list(episode_numbers), list(all_critic_losses))

    # 3. Automatically redraw the bounds of the graphs set
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()

    # 4. Redraw the figure and pause briefly for the interface to update.
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)  # This is very important!

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()
plt.ioff()
plt.show()

