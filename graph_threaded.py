import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading
import time

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

def evaluate_agent(env, actor_model, eval_episodes=10):
    """
    It tests the agent without learning, using only the actions it knows best at the time.
    """
    # eval_env = gym.make("CartPole-v1", render_mode='human')
    total_rewards = 0

    for _ in range(eval_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Speed up by telling PyTorch not to calculate gradients
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = actor_model(state_tensor)

                # Instead of random selection (.sample()), choose the action with the highest probability
                action = torch.argmax(action_probs, dim=1)

            next_state, reward, done, truncated, info = env.step(action.item())
            episode_reward += reward
            state = next_state
        total_rewards += episode_reward

    # env.close()
    return total_rewards / eval_episodes

def update_annot(line, annot, ind):
    """Show info the hovered point."""
    x, y = line.get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    text = f"Episode: {int(x[ind['ind'][0]])}\nValue: {y[ind['ind'][0]]:.2f}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.7)

def on_hover(event):
    """Main function that called when mouse moved."""
    vis = annot.get_visible()
    # If the mouse over a line...
    if event.inaxes in [ax1, ax2]:
        lines_to_check = []
        target_annot = None
        if event.inaxes == ax1:
            lines_to_check = [line_rewards, line_eval]
            target_annot = annot
        elif event.inaxes == ax2:
            lines_to_check = [line_actor_loss, line_critic_loss]
            target_annot = annot2

        for line in lines_to_check:
            cont, ind = line.contains(event)
            if cont:
                update_annot(line, target_annot, ind)
                target_annot.set_visible(True)
                fig.canvas.draw_idle()
                return
    # If mouse not hover the line hide the note.
    if vis:
        annot.set_visible(False)
        annot2.set_visible(False)
        fig.canvas.draw_idle()

# --- Global Variables for Evaluation ---
EVAL_LOCK = threading.Lock() # Lock to prevent concurrent access to variables
latest_eval_score = 0
is_eval_done = True

# --- VISUALIZATION SETUP ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("Actor-Critical Training Process")
fig.canvas.mpl_connect("motion_notify_event", on_hover)

# Chart 1: Awards
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Rewards")
ax1.set_title("Total Rewards per Episode")
line_rewards, = ax1.plot([], [], label="Education Award", color="cyan", picker=True, pickradius=5)
line_eval, = ax1.plot([], [], label="Evaluation Award", color="purple", linewidth=2, marker='o', picker=True, pickradius=5)
ax1.legend()

# Graph 2: Losses
ax2.set_xlabel("Episode")
ax2.set_ylabel("Loss")
ax2.set_title("Actor and Critical Losses")
line_actor_loss, = ax2.plot([], [], label="Actor Loss", color="blue", picker=True, pickradius=5)
line_critic_loss, = ax2.plot([], [], label="Critical Loss", color="red", picker=True, pickradius=5)
ax2.legend()

# Create an annotation box that will display information when hovered over.
annot = ax1.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)
# We will use the same annotation box for both graphs.
annot2 = ax2.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot2.set_visible(False)

episode_numbers = deque(maxlen=100)
all_rewards = deque(maxlen=100)
all_actor_losses = deque(maxlen=100)
all_critic_losses = deque(maxlen=100)

eval_episode_numbers = []
all_eval_rewards = deque(maxlen=100)

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

    if (episode + 1) % 10 == 0:
        avg_eval_reward = evaluate_agent(env, actor, eval_episodes=10)
        eval_episode_numbers.append(episode)
        all_eval_rewards.append(avg_eval_reward)
        print(f"--- Evaluation Result: Average Reward = {avg_eval_reward:.2f} ---")

    # 2. Update line data
    line_rewards.set_data(list(episode_numbers), list(all_rewards))
    line_actor_loss.set_data(list(episode_numbers), list(all_actor_losses))
    line_critic_loss.set_data(list(episode_numbers), list(all_critic_losses))
    line_eval.set_data(eval_episode_numbers, list(all_eval_rewards))

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

