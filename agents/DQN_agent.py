import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from agents import BaseAgent

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.3):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-6

    def push(self, state, action, reward, next_state, done, td_error=None):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        if td_error is not None:
            prio = (abs(td_error) + self.epsilon) ** self.alpha
        else:
            if self.tree.n_entries == 0:
                prio = self.epsilon ** self.alpha
            else:
                prio = np.max(self.tree.tree[-self.tree.capacity:])
        prio = prio if prio > 0 else self.epsilon**self.alpha
        self.tree.add(prio, (state, action, reward, next_state, done))

    def sample(self, batch_size, beta=0.4):
        if random.random() < 0.1:
            data_idxs = np.random.choice(self.tree.n_entries, batch_size, replace=False)
            batch = [self.tree.data[i] for i in data_idxs]
            idxs = [i + self.tree.capacity - 1 for i in data_idxs]
            is_weights = np.ones(batch_size, dtype=np.float32)
            states, actions, rewards, next_states, dones = zip(*batch)
            return (torch.stack(states), actions, rewards, torch.stack(next_states),
                    dones, idxs, is_weights)

        batch, idxs, priorities = [], [], []
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, prio, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(prio)
        sampling_probabilities = np.array(priorities) / self.tree.total
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states), actions, rewards, torch.stack(next_states),
                dones, idxs, is_weights)

    def update_priorities(self, idxs, td_errors):
        td_errors = np.clip(td_errors, -1.0, 1.0)
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, prio in zip(idxs, priorities):
            self.tree.update(idx, prio)

    def __len__(self):
        return self.tree.n_entries


class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        target_update: int = 10,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 250_000,
        per_alpha: float = 0.3,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 200_000,
        failure_weight: float = 0.1
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scaler = GradScaler()

        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=per_alpha)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.action_dim = action_dim
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.failure_weight = failure_weight

    def select_action(self, state: np.ndarray) -> int:
        self.policy_net.eval()
        eps_threshold = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() < eps_threshold:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def take_action(self, state: tuple[float, float, float]) -> int:
        return self.select_action(np.array(state, dtype=np.float32))

    def store_transition(self, state, action, reward, next_state, done: bool):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        beta = min(1.0, self.per_beta_start + (1.0 - self.per_beta_start) *
                    (self.steps_done / self.per_beta_frames))

        states_t, actions, rewards, next_states_t, dones, idxs, is_weights = \
            self.replay_buffer.sample(self.batch_size, beta)

        states_t = states_t.to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = next_states_t.to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        is_weights_t = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        self.policy_net.train()
        with autocast():
            current_q = self.policy_net(states_t).gather(1, actions_t)
            with torch.no_grad():
                next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

            weight_factor = dones_t + self.failure_weight * (1 - dones_t)
            loss = (is_weights_t * weight_factor * (current_q - target_q).pow(2)).mean()

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        td_errors = (target_q - current_q).detach().cpu().numpy().squeeze()
        self.replay_buffer.update_priorities(idxs, td_errors)

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def finalize_training(self):
        self.epsilon_start = 0.0
        self.epsilon_end = 0.0
        self.steps_done = self.epsilon_decay

    def update(self, state, reward, action):
        pass
