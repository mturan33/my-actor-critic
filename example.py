import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

# --- Hyperparameters ---
LEARNING_RATE = 0.001
GAMMA = 0.99
MAX_EPISODES = 2000


# --- Model Definitions ---

class Actor(nn.Module):
    """Actor network that outputs a policy (action probabilities)."""

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
    """Critic network that evaluates a state and outputs its value."""

    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.network(state)


# --- Initialization ---
# Use 'human' render_mode to see the simulation window
env = gym.make("CartPole-v1", render_mode='human')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

actor = Actor(state_size, action_size)
critic = Critic(state_size)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)

# --- Main Training Loop ---
for episode in range(MAX_EPISODES):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Get action probabilities from the actor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = actor(state_tensor)

        # Create a distribution and sample an action
        dist = Categorical(action_probs)
        action = dist.sample()

        # Perform the action in the environment
        next_state, reward, done, truncated, info = env.step(action.item())

        # Calculate the advantage (TD Error): R + gamma * V(s') - V(s)
        # This measures how much better the action taken was than expected.
        current_state_value = critic(state_tensor)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # The value of the terminal state is 0.
        if done:
            next_state_value = torch.tensor([0.0])
        else:
            next_state_value = critic(next_state_tensor)

        advantage = reward + GAMMA * next_state_value - current_state_value

        # --- Update Critic ---
        # The critic's goal is to minimize the advantage (TD Error).
        critic_loss = advantage.pow(2)  # Mean Squared Error Loss

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # --- Update Actor ---
        # The actor's goal is to take actions that lead to a high advantage.
        # We use .detach() on advantage to prevent its gradients from flowing into the critic.
        log_prob = dist.log_prob(action)
        actor_loss = -log_prob * advantage.detach()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Move to the next state
        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()