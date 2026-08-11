import torch
import torch.nn as nn

class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Soft Actor-Critic Eleştirmen (Critic) Ağı
        Hem uçağın durumunu (State) hem de ajanın hamlesini (Action) alıp 
        o hamlenin taktiksel değerini (Q-Value) puanlar.
        """
        super(SACCritic, self).__init__()
        
        # --- 1. Q-Ağı (Eleştirmen 1) ---
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Tek bir puan (Q-değeri) döndürür
        )
        
        # --- 2. Q-Ağı (Eleştirmen 2) ---
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        # Durum ve Aksiyon vektörlerini yan yana birleştir
        # Örneğin: 11 boyutlu state + 3 boyutlu action = 14 boyutlu girdi
        x = torch.cat([state, action], dim=-1)
        
        # İki ağdan da bağımsız puanlamaları al
        q1_value = self.q1(x)
        q2_value = self.q2(x)
        
        return q1_value, q2_value