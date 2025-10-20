import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym

# OYUNCU (AKTÖR)
# Gözlemi (state) alır, her bir eylem için bir olasılık döndürür.
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1) # Çıktıyı olasılıklara dönüştürür
        )

    def forward(self, state):
        return self.network(state)

# ANTRENÖR (KRİTİK)
# Gözlemi (state) alır, o durumun ne kadar "iyi" olduğunu tahmin eden tek bir sayı döndürür.
class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Çıktı tek bir değer: State Value
        )

    def forward(self, state):
        return self.network(state)

# --- Hiperparametreler ---
env = gym.make("CartPole-v1", render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

actor = Actor(state_size, action_size)
critic = Critic(state_size)

# Optimizer'lar, ağların nasıl öğreneceğini yönetir
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)

gamma = 0.99  # Gelecekteki ödüllerin ne kadar önemli olduğu (İndirgeme Faktörü)

# --- Eğitim Döngüsü ---
for episode in range(1000):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # 1. Aktör oynar
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = actor(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()  # Olasılıklara göre bir eylem seç

        # 2. Çevre tepki verir
        next_state, reward, done, truncated, info = env.step(action.item())

        total_reward += reward

        # 3. Antrenör (Kritik) değerlendirir
        current_state_value = critic(state_tensor)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        next_state_value = critic(next_state_tensor)

        if done:  # Eğer bölüm bittiyse, bir sonraki durumun değeri 0'dır
            next_state_value = torch.tensor([0.0])

        # 4. "Beklenti Farkı"nı (TD Error) hesapla
        # Bu, öğrenmenin temel sinyalidir!
        advantage = reward + gamma * next_state_value - current_state_value

        # 5. Aktör'ü güncelle
        log_prob = dist.log_prob(action)
        actor_loss = -log_prob * advantage.detach()  # Kritik'in hatasını Aktör'e bulaştırma (.detach())

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # 6. Kritik'i güncelle
        critic_loss = advantage.pow(2)  # Kritik, hatasını (advantage'ı) minimize etmeye çalışır

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Bir sonraki adıma geç
        state = next_state

    print(f"Episode {episode}: Total Reward: {total_reward}")

env.close()