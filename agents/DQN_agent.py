import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents import BaseAgent

class QNetwork(nn.Module):
    '''A 3-layer MLP to approximate the Q function.'''
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
    '''The main class for the DQN agent.'''
    def __init__(
        self,
        state_dim: int = 3,           # Dimension of a state; currently states are described as (x, y, phi), so state_dim = 3
        action_dim: int = 3,          # Dimension of the action space; currently the robot can take 3 different actions, so action_dim = 3.
        buffer_capacity: int = 10000, # How many transitions are stored in the replay buffer
        batch_size: int = 256,        # Training batches will contain this many steps.
        gamma: float = 0.99,          # Discount factor for the return.
        lr: float = 1e-3,             # Optimizer learning rate.
        target_update: int = 500,     # How many steps between each update of the target network.
        epsilon_start: float = 1.0,   # The exploration probability epsilon decays from epsilon_start
        epsilon_end: float = 0.01,    # to epsilon_end
        epsilon_decay: int = 250_000, # across epsilon_decay steps.
        success_frac: float = 0.5     # Fraction of each training batch that is drawn only from successful episodes.
    ):
        # Initialize policy and target nets
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Use separate replay buffers for succesful episodes (reached the target) and failed ones
        self.success_replay = []  # Format: (state, action, reward, next_state, done, ep_length)
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
        '''Select a random action with probability eps, 
        or the best action according to the policy net otherwise.'''
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
        """Save the given transition in the buffer for the current episode."""
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

        # 3) put samples together into a batch
        batch = success_batch + fail_batch
        random.shuffle(batch)

        states, actions, rewards, next_states, dones, _ = zip(*batch)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 4) calculated predicted and target q
        current_q = self.policy_net(states_t).gather(1, actions_t)
        next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0].detach()
        target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        # 5) compute loss & do backprop
        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Copy policy net to target net every target_update steps
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def finalize_training(self):
        # turn off exploration
        self.epsilon_start = 0.0
        self.epsilon_end   = 0.0

