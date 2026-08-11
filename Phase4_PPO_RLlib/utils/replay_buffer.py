import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim=14, action_dim=3, capacity=1000000):
        """
        A ve B (DMD) Matrislerini de depolayan Özel Hafıza (Replay Buffer)
        Performans için listeler (append) yerine, önceden ayrılmış Numpy array'leri kullanıyoruz.
        """
        self.capacity = capacity
        self.ptr = 0  # Hafıza yazma işaretçisi
        self.size = 0 # Mevcut dolu kapasite
        
        # Standart RL Verileri
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # FAZ 3'E ÖZEL: DMD Aerodinamik Matrisleri
        self.As = np.zeros((capacity, state_dim, state_dim), dtype=np.float32)
        self.Bs = np.zeros((capacity, state_dim, action_dim), dtype=np.float32)

    def push(self, state, action, reward, next_state, done, A, B):
        """ Yeni bir deneyimi hafızaya yazar (Eskilerin üzerine yazarak / FIFO) """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.As[self.ptr] = A
        self.Bs[self.ptr] = B
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """ Eğitim için rastgele bir yığın (batch) çeker ve PyTorch Tensörüne çevirir """
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[ind]),
            torch.FloatTensor(self.actions[ind]),
            torch.FloatTensor(self.rewards[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.dones[ind]),
            torch.FloatTensor(self.As[ind]),  # Kalkanın gradient hesabı için A matrisleri
            torch.FloatTensor(self.Bs[ind])   # Kalkanın gradient hesabı için B matrisleri
        )

    def __len__(self):
        return self.size