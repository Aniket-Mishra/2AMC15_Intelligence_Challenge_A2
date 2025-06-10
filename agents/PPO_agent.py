import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents import BaseAgent

# PPO has actor-critic framework so we make two networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# PPO memory to store transitions
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def store(self, state, action, logprob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

# the PPO agent class
class PPOAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=3e-4,
        clip_epsilon=0.2,
        update_epochs=15,
        batch_size=64,
        gae_lambda=0.95
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.memory = PPOMemory()
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.action_dim = action_dim

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob.item()

    def take_action(self, state):
        action, logprob = self.select_action(np.array(state, dtype=np.float32))
        return action

    def store_transition(self, state, action, logprob, reward, done):
        self.memory.store(state, action, logprob, reward, done)

    # Generalized Advantage Estimation algorithm for calculating advantages
    def compute_gae(self, rewards, dones, values, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return advantages
    
    def learn(self):
        if len(self.memory.states) < self.batch_size:
            return

        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_logprobs = torch.FloatTensor(self.memory.logprobs).to(self.device)
        rewards = self.memory.rewards
        dones = self.memory.dones

        with torch.no_grad():
            values = self.critic(states).squeeze().cpu().numpy().tolist()
            next_state = torch.FloatTensor(np.array(self.memory.states[-1])).unsqueeze(0).to(self.device)
            next_value = self.critic(next_state).squeeze().cpu().numpy() 
        advantages = self.compute_gae(rewards, dones, values, next_value)
        returns = [adv + val for adv, val in zip(advantages, values)]
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        for _ in range(self.update_epochs):
            idx = np.arange(len(states))
            np.random.shuffle(idx)
            for start in range(0, len(states), self.batch_size):
                # create batches
                end = start + self.batch_size
                batch_idx = idx[start:end]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                probs = self.actor(batch_states)
                dist = torch.distributions.Categorical(probs)
                logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                # PPO loss calculation
                # calculate how much the policy has changed
                ratios = torch.exp(logprobs - batch_old_logprobs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                # actor loss is the minimum of the two surrogate losses
                actor_loss = -torch.min(surr1, surr2).mean()
                # critic loss is mse between critic values and returns
                critic_loss = nn.MSELoss()(self.critic(batch_states).squeeze(-1), batch_returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory.clear()

    def finalize_training(self): 
        pass

    def update(self, state, reward, action): # i think we can delete this but needs to be deleted from base agent?
        pass