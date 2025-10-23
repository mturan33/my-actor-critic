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
# 2. DEFINING THE STREAMLIT INTERFACE
# -----------------------------------------------------------------------------

# Set page title and layout
st.set_page_config(layout="wide")
st.title("🤖 Live Actor-Critic RL Training: CartPole")

st.markdown("""
This interactive web application demonstrates a Reinforcement Learning agent learning to solve the **CartPole-v1** problem from scratch, right in your browser.
The agent uses a simple **Actor-Critic** algorithm.

- **Simulation Screen:** Shows the agent's performance in real-time.
- **Training Charts:** Visualize the agent's learning process.
- **Training Logs:** Displays the raw data for the last 10 episodes.
- **Github Repository:** https://github.com/mturan33/my-actor-critic
""")

st.info("Click the 'Start Training!' button below to begin the live training process.")

# Divide the page into two main columns
col1, col2 = st.columns(2)

with col1:
    st.header("🎮 Simulation Screen")
    # This 'st.empty()' will act as a placeholder for the simulation image.
    simulation_image_placeholder = st.empty()

with col2:
    st.header("📊 Training Charts")
    charts_placeholder = st.empty()

st.header("📜 Training Logs")
log_placeholder = st.empty()

# "Start Training" button
st.markdown("""
<style>
div.stButton > button:first-child {
    font-size: 20px;
    height: 3em;
    width: 100%;
    background-color: #008CBA; /* Mavi renk */
    color: white;
    border-radius: 10px;
}
</style>""", unsafe_allow_html=True)

if 'training_started' not in st.session_state:
    st.session_state['training_started'] = False

if st.button("🚀 Start Training!"):
    st.session_state['training_started'] = True

if st.session_state['training_started']:
    # --- Hyperparameters ---
    LEARNING_RATE = 0.0005
    GAMMA = 0.98
    MAX_EPISODES = 2000

    # --- Environment and Model Setup ---
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    actor = Actor(state_size, action_size)
    critic = Critic(state_size)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    # --- Data storage for charts ---
    chart_data = pd.DataFrame(columns=["Episode", "Reward", "Actor Loss", "Critic Loss", "Evaluation Reward"])

    # --- Main Training Loop ---
    for episode in range(MAX_EPISODES):
        state, info = env.reset()
        done = False

        episode_reward = 0
        episode_actor_losses = []
        episode_critic_losses = []

        # This loop runs for every single step WITHIN an episode
        while not done:
            # --- 1. ACTION SELECTION ---
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = actor(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()

            # --- 2. INTERACTING WITH THE ENVIRONMENT ---
            frame = env.render()
            next_state, reward, done, truncated, info = env.step(action.item())

            # --- 3. CALCULATING THE LEARNING SIGNAL ---
            # ... (advantage calculation logic is correct) ...
            current_state_value = critic(state_tensor)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            if done:
                next_state_value = torch.tensor([0.0])
            else:
                next_state_value = critic(next_state_tensor)
            advantage = reward + GAMMA * next_state_value - current_state_value

            # --- 4. NETWORK UPDATE (LEARNING MOMENT) ---
            # ... (critic and actor update logic is correct) ...
            critic_loss = advantage.pow(2)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            log_prob = dist.log_prob(action)
            actor_loss = -log_prob * advantage.detach()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # --- Data Collection (for this step) ---
            episode_reward += reward
            episode_actor_losses.append(actor_loss.item())
            episode_critic_losses.append(critic_loss.item())

            state = next_state

            # --- VISUALIZATION (Streamlit Method) ---
            # Update the simulation screen every * steps
            if int(episode_reward) % 10 == 0:
                width = int(frame.shape[1] * 0.50)
                height = int(frame.shape[0] * 0.50)
                resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                simulation_image_placeholder.image(resized_frame, caption=f"Episode: {episode + 1} | Step: {int(episode_reward)}")

        # --- END OF EPISODE UPDATES ---

        # 1. Calculate the evaluation score (if it's time)
        eval_reward = np.nan
        if (episode + 1) % 20 == 0:
            eval_reward = evaluate_agent(actor, eval_episodes=5)

        # 2. Aggregate ALL data for this completed episode
        new_data_row = {
            "Episode": episode + 1,
            "Reward": episode_reward,
            "Evaluation Reward": eval_reward,
            "Actor Loss": np.mean(episode_actor_losses),
            "Critic Loss": np.mean(episode_critic_losses)
        }

        # 3. Append the single summary row to the DataFrame
        chart_data = pd.concat([chart_data, pd.DataFrame([new_data_row])], ignore_index=True)

        # 4. Calculate moving averages AFTER the new data is added
        if (episode + 1) % 30 == 0:
            chart_data['Reward Trend'] = chart_data['Reward'].rolling(window=10, min_periods=1).mean()
            chart_data['Actor Loss Trend'] = chart_data['Actor Loss'].rolling(window=10, min_periods=1).mean()
            chart_data['Critic Loss Trend'] = chart_data['Critic Loss'].rolling(window=10, min_periods=1).mean()

            # 5. Update the chart placeholder using the complete and correct data
            with charts_placeholder.container():

                g_col1, g_col2 = st.columns(2)

                with g_col1:
                    # --- Reward Chart ---
                    st.subheader("Reward Chart")

                    # Melt the data for BOTH reward types into a long format
                    reward_data_long = chart_data.melt(
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
                    loss_data_long = chart_data.melt(id_vars=["Episode"], value_vars=["Actor Loss Trend", "Critic Loss Trend"],
                                                     var_name="Loss Type", value_name="Value")
                    loss_chart = alt.Chart(loss_data_long).mark_line().encode(
                        x=alt.X('Episode:Q', axis=alt.Axis(title='Episode')), y=alt.Y('Value:Q', axis=alt.Axis(title='Loss')),
                        color=alt.Color('Loss Type:N', legend=alt.Legend(title="Loss Type")),
                        tooltip=['Episode', 'Value']).properties(
                        title="Actor vs. Critic Loss").interactive()
                    st.altair_chart(loss_chart, use_container_width=True)

            # 6. Update the log placeholder
            with log_placeholder.container():
                st.text(f"Episode {episode + 1}: Total Reward: {episode_reward}")
                if not np.isnan(eval_reward):
                    st.success(f"--- Evaluation Result (Episode {episode+1}): Average Reward = {eval_reward:.2f} ---")
                st.dataframe(chart_data.tail(10))

    env.close()
    st.success("Training complete!")