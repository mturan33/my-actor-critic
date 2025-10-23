import streamlit as st
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import time
import altair as alt
import cv2

# How To Run In Local: powershell streamlit run app.py

# -----------------------------------------------------------------------------
# 1. ACTOR-CRITIC MODELS
# -----------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
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
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.network(state)

def evaluate_agent(actor_model, eval_episodes=5):
    eval_env = gym.make("CartPole-v1")
    total_rewards = 0

    for _ in range(eval_episodes):
        state, info = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = actor_model(state_tensor)
                action = torch.argmax(action_probs, dim=1)

            next_state, reward, done, truncated, info = eval_env.step(action.item())
            episode_reward += reward
            state = next_state
        total_rewards += episode_reward

    eval_env.close()
    return total_rewards / eval_episodes
# -----------------------------------------------------------------------------
# 2. HELPER FUNCTION TO RUN ONE EPISODE
# -----------------------------------------------------------------------------
def run_one_episode(env, actor, critic, actor_optimizer, critic_optimizer, gamma):
    """Runs a single episode of training and returns all the summary data."""
    state, info = env.reset()
    done = False

    episode_reward = 0
    episode_actor_losses = []
    episode_critic_losses = []

    last_frame = None

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = actor(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()

        frame = env.render()
        last_frame = frame  # Store the latest frame
        next_state, reward, done, truncated, info = env.step(action.item())

        current_state_value = critic(state_tensor)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        if done:
            next_state_value = torch.tensor([0.0])
        else:
            next_state_value = critic(next_state_tensor)

        advantage = reward + gamma * next_state_value - current_state_value

        critic_loss = advantage.pow(2)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        log_prob = dist.log_prob(action)
        actor_loss = -log_prob * advantage.detach()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        episode_reward += reward
        episode_actor_losses.append(actor_loss.item())
        episode_critic_losses.append(critic_loss.item())
        state = next_state

    return {
        "reward": episode_reward,
        "actor_loss": np.mean(episode_actor_losses),
        "critic_loss": np.mean(episode_critic_losses),
        "last_frame": last_frame
    }
# -----------------------------------------------------------------------------
# 3. STREAMLIT UI AND STATE MANAGEMENT
# -----------------------------------------------------------------------------
# --- Page Configuration ---
st.set_page_config(layout="wide")
st.title("🤖 Live Actor-Critic RL Training: CartPole")
st.markdown(
    "This app demonstrates an RL agent learning from scratch. The training loop is managed by Streamlit's Session State for a responsive UI.")

# --- Initialize Session State (The App's Memory) ---
# This part runs only ONCE, when the app is first loaded.
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
    st.session_state.run_training = False

    # --- Hyperparameters ---
    st.session_state.learning_rate = 0.0005
    st.session_state.gamma = 0.98

    # --- Environment and Models ---
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    st.session_state.env = env
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    st.session_state.actor = Actor(state_size, action_size)
    st.session_state.critic = Critic(state_size)
    st.session_state.actor_optimizer = torch.optim.Adam(st.session_state.actor.parameters(),
                                                        lr=st.session_state.learning_rate)
    st.session_state.critic_optimizer = torch.optim.Adam(st.session_state.critic.parameters(),
                                                         lr=st.session_state.learning_rate)

    # --- Data Storage ---
    st.session_state.chart_data = pd.DataFrame(
        columns=["Episode", "Reward", "Evaluation Reward", "Actor Loss", "Critic Loss"])
    st.session_state.episode_count = 0

# --- UI Layout ---
col1, col2 = st.columns([1, 2])  # Make the simulation smaller, charts bigger

with col1:
    st.header("🎮 Simulation")
    simulation_image_placeholder = st.empty()
    # Show the initial state of the environment
    if not st.session_state.training_started:
        st.session_state.env.reset()
        simulation_image_placeholder.image(st.session_state.env.render(), caption="Ready to Train!")

with col2:
    st.header("📊 Training Charts")
    charts_placeholder = st.empty()
    st.header("📜 Training Logs")
    log_placeholder = st.empty()

# --- Control Buttons ---
c1, c2 = st.columns(2)
if c1.button("▶️ Start / Pause Training", use_container_width=True):
    st.session_state.run_training = not st.session_state.run_training

if c2.button("🔄 Reset Training", use_container_width=True):
    # This resets the entire state, forcing a re-initialization
    st.session_state.clear()
    st.rerun()

# --- The Main Loop (Streamlit Style) ---
if st.session_state.run_training:
    st.session_state.training_started = True

    # Run one episode
    episode_data = run_one_episode(
        st.session_state.env,
        st.session_state.actor,
        st.session_state.critic,
        st.session_state.actor_optimizer,
        st.session_state.critic_optimizer,
        st.session_state.gamma
    )
    st.session_state.episode_count += 1

    # --- Evaluation ---
    eval_reward = np.nan
    if st.session_state.episode_count % 10 == 0:
        eval_reward = evaluate_agent(st.session_state.actor, eval_episodes=5)

    # --- Update Data ---
    new_data_row = {
        "Episode": st.session_state.episode_count,
        "Reward": episode_data["reward"],
        "Evaluation Reward": eval_reward,
        "Actor Loss": episode_data["actor_loss"],
        "Critic Loss": episode_data["critic_loss"]
    }
    st.session_state.chart_data = pd.concat([st.session_state.chart_data, pd.DataFrame([new_data_row])], ignore_index=True)

    # Calculate moving averages AFTER the new data is added
    st.session_state.chart_data['Reward Trend'] = st.session_state.chart_data['Reward'].rolling(window=10, min_periods=1).mean()
    st.session_state.chart_data['Actor Loss Trend'] = st.session_state.chart_data['Actor Loss'].rolling(window=10, min_periods=1).mean()
    st.session_state.chart_data['Critic Loss Trend'] = st.session_state.chart_data['Critic Loss'].rolling(window=10, min_periods=1).mean()
    # --- Update UI Elements ---
    simulation_image_placeholder.image(episode_data["last_frame"], caption=f"Episode: {st.session_state.episode_count} | Reward: {episode_data['reward']:.0f}")

    # --- Update Charts ---
    with charts_placeholder.container():

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            # --- Reward Chart ---
            st.subheader("Reward Chart")

            # Melt the data for BOTH reward types into a long format
            reward_data_long = st.session_state.chart_data.melt(
                id_vars=["Episode"],
                value_vars=["Reward Trend", "Evaluation Reward"],
                var_name="Data Type",
                value_name="Value"
            )

            # Create a single chart with two different marks (line and point)
            reward_chart = alt.Chart(reward_data_long).encode(
                x=alt.X('Episode:Q', axis=alt.Axis(title='Episode')),
                y=alt.Y('Value:Q', axis=alt.Axis(title='Total Reward')),
                color=alt.Color('Data Type:N', legend=alt.Legend(title="Data Type"))
            )

            final_reward_chart = (reward_chart.transform_filter(
                alt.datum['Data Type'] == 'Reward Trend'
            ).mark_line() + reward_chart.transform_filter(
                alt.datum['Data Type'] == 'Evaluation Reward'
            ).mark_point(filled=True, size=80, opacity=0.8)).interactive().properties(
                title="Training Trend (Moving Avg) vs. Evaluation Score"
            )

            st.altair_chart(final_reward_chart, use_container_width=True)

        with g_col2:
            st.subheader("Loss Charts")
            loss_data_long = st.session_state.chart_data.melt(id_vars=["Episode"], value_vars=["Actor Loss Trend", "Critic Loss Trend"],
                                             var_name="Loss Type", value_name="Value")
            loss_chart = alt.Chart(loss_data_long).mark_line().encode(
                x=alt.X('Episode:Q', axis=alt.Axis(title='Episode')), y=alt.Y('Value:Q', axis=alt.Axis(title='Loss')),
                color=alt.Color('Loss Type:N', legend=alt.Legend(title="Loss Type")),
                tooltip=['Episode', 'Value']).properties(
                title="Actor vs. Critic Loss").interactive()
            st.altair_chart(loss_chart, use_container_width=True)

        # 6. Update the log placeholder
        with log_placeholder.container():
            st.text(f"Current Episode: {st.session_state.episode_count}")
            if not np.isnan(eval_reward):
                st.success(f"--- Evaluation Result: Avg Reward = {eval_reward:.2f} ---")
            st.dataframe(st.session_state.chart_data.tail(10))
    # --- Trigger the next run ---
    # This is a small "hack" to make Streamlit re-run the script automatically
    # creating a continuous loop without a `while True`.
    time.sleep(0.1) # Give the UI a moment to breathe
    st.rerun()