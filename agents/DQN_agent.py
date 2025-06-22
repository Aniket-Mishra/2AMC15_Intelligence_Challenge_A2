import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_capacity: int = 10000,
        batch_size: int = 256,
        gamma: float = 0.99,
        lr: float = 1e-3,
        target_update: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 250_000,
        success_frac: float = 0.5  # fraction of each batch drawn from successful episodes
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # two separate buffers
        self.success_replay = []  # (state, action, reward, next_state, done, ep_length)
        self.fail_replay    = []  # same format
        self.current_episode = []

        self.capacity = buffer_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.success_frac = success_frac
        self.steps_done = 0
        self.action_dim = action_dim

    def start_episode(self):
        """Clear per-episode buffer."""
        self.current_episode.clear()

    def select_action(self, state: np.ndarray) -> int:
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
              np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def take_action(self, state: tuple[float, float, float]) -> int:
        return self.select_action(np.array(state, dtype=np.float32))

    def record_transition(self, state, action, reward, next_state, done: bool):
        """Buffer into current episode."""
        self.current_episode.append((state, action, reward, next_state, done))

    def end_episode(self, success: bool):
        """At episode end, push transitions into success or failure buffer."""
        buf = self.success_replay if success else self.fail_replay
        ep_len = len(self.current_episode)
        for entry in self.current_episode:
            if len(buf) >= self.capacity:
                buf.pop(0)
            buf.append((*entry, ep_len))
        self.current_episode.clear()

    def learn(self):
        # need enough total to sample from both
        if len(self.success_replay) + len(self.fail_replay) < self.batch_size:
            return

        # determine counts
        n_success = int(self.batch_size * self.success_frac)
        n_fail    = self.batch_size - n_success

        # clamp to available transitions
        avail_succ = len(self.success_replay)
        avail_fail = len(self.fail_replay)
        n_success = min(n_success, avail_succ)
        n_fail = min(n_fail, avail_fail)
        if n_success + n_fail == 0:
            return

        # 1) sample successes, allow replacement if too few
        success_batch = []
        if n_success > 0 and self.success_replay:
            lengths = np.array([e[5] for e in self.success_replay], dtype=np.float32)
            weights = 1.0 / lengths
            probs = weights / weights.sum()
            replace_succ = (n_success > avail_succ)
            idxs = np.random.choice(avail_succ, n_success, p=probs, replace=replace_succ)
            success_batch = [self.success_replay[i] for i in idxs]

        # 2) sample failures, allow replacement if too few
        fail_batch = []
        if n_fail > 0 and self.fail_replay:
            replace_fail = (n_fail > avail_fail)
            idxs = np.random.choice(avail_fail, n_fail, replace=replace_fail)
            fail_batch = [self.fail_replay[i] for i in idxs]

        batch = success_batch + fail_batch
        random.shuffle(batch)

        states, actions, rewards, next_states, dones, _ = zip(*batch)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t)
        next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0].detach()
        target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def finalize_training(self):
        # turn off exploration
        self.epsilon_start = 0.0
        self.epsilon_end   = 0.0

