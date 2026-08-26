"""Preference-conditioned one-step PPO with one scalar critic.

This is the fair single-critic counterpart of the project's multi-critic PPO:
the actor architecture and preference input are retained, while the critic is
trained on the preference-weighted scalar reward.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Buffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.preferences = [], [], []
        self.dones, self.auxiliary_rewards = [], []


class SingleCriticActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, num_obj: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim + num_obj, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )
        self.actor_log_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.actor_log_std[-1].bias, -1.0)

    def forward(self, state, preference):
        features = self.shared(torch.cat([state, preference], dim=-1))
        mean = self.actor_mean(features) * 3.0
        std = torch.exp(torch.clamp(self.actor_log_std(features), -5.0, 2.0))
        return mean, std, self.critic(features)


class TrajectorySingleCriticPPOAgent:
    def __init__(
        self, state_dim, action_dim, num_obj=2, lr=3e-4, gamma=0.99,
        gae_lambda=0.95, ppo_clip=0.2, ppo_epochs=4, entropy_coef=0.01,
        value_loss_coef=0.5, mini_batch_size=32, max_grad_norm=0.5,
        device=None,
    ):
        self.num_obj = num_obj
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = SingleCriticActorCritic(state_dim, action_dim, num_obj).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = Buffer()
        self.auxiliary_actor_coef = 0.0

    @staticmethod
    def _standardize(values):
        return (values - values.mean()) / (values.std(unbiased=False) + 1e-8)

    def select_action(self, state, preference=None):
        if preference is None:
            preference = np.ones(self.num_obj, dtype=np.float32) / self.num_obj
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            pref_t = torch.as_tensor(preference, dtype=torch.float32, device=self.device).unsqueeze(0)
            mean, std, value = self.policy(state_t, pref_t)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            entropy = dist.entropy().sum(-1, keepdim=True)
        return (
            action.squeeze(0).cpu().numpy(), float(log_prob.item()),
            value.squeeze(0).cpu().numpy(), float(entropy.item()),
        )

    def store_transition_multi(
        self, state, action, rewards, log_prob, values, done,
        entropy=0.0, preference=None, auxiliary_reward=0.0,
    ):
        pref = np.asarray(preference, dtype=np.float32)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(float(np.dot(np.asarray(rewards, dtype=np.float32), pref)))
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(float(np.asarray(values).reshape(-1)[0]))
        self.buffer.preferences.append(pref)
        self.buffer.dones.append(bool(done))
        self.buffer.auxiliary_rewards.append(float(auxiliary_reward))

    def _compute_trajectory_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        for step in range(len(rewards) - 1, -1, -1):
            nonterminal = 1.0 - dones[step]
            next_value = (
                values[step + 1]
                if step + 1 < len(values)
                else torch.zeros_like(values[step])
            )
            delta = rewards[step] + self.gamma * next_value * nonterminal - values[step]
            running = delta + self.gamma * self.gae_lambda * nonterminal * running
            advantages[step] = running
        return advantages

    def _discount_auxiliary_rewards(self, rewards, dones):
        advantages = torch.zeros_like(rewards)
        running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        for step in range(len(rewards) - 1, -1, -1):
            nonterminal = 1.0 - dones[step]
            running = rewards[step] + self.gamma * self.gae_lambda * nonterminal * running
            advantages[step] = running
        return advantages

    def update(self, memory):
        if len(memory.states) < self.mini_batch_size:
            return 0.0
        states = torch.as_tensor(np.asarray(memory.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(memory.actions), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(np.asarray(memory.rewards), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(memory.log_probs), dtype=torch.float32, device=self.device).reshape(-1, 1)
        old_values = torch.as_tensor(np.asarray(memory.values), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.asarray(memory.dones), dtype=torch.float32, device=self.device)
        preferences = torch.as_tensor(np.asarray(memory.preferences), dtype=torch.float32, device=self.device)
        auxiliary_rewards = torch.as_tensor(
            np.asarray(memory.auxiliary_rewards), dtype=torch.float32, device=self.device
        )
        raw_advantages = self._compute_trajectory_gae(rewards, old_values.detach(), dones)
        advantages = self._standardize(raw_advantages)
        if self.auxiliary_actor_coef != 0.0 and auxiliary_rewards.numel():
            advantages = advantages + self.auxiliary_actor_coef * self._standardize(
                self._discount_auxiliary_rewards(auxiliary_rewards, dones)
            )
            advantages = self._standardize(advantages)
        returns = raw_advantages + old_values.detach()
        total_loss, updates = 0.0, 0
        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(len(states), device=self.device)
            for start in range(0, len(states), self.mini_batch_size):
                index = permutation[start:start + self.mini_batch_size]
                mean, std, values = self.policy(states[index], preferences[index])
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(actions[index]).sum(-1, keepdim=True)
                ratio = torch.exp(log_probs - old_log_probs[index])
                advantage = advantages[index].reshape(-1, 1)
                actor_loss = -torch.minimum(
                    ratio * advantage,
                    torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantage,
                ).mean()
                critic_loss = nn.functional.mse_loss(values.reshape(-1), returns[index])
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * dist.entropy().sum(-1).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += float(loss.item())
                updates += 1
        return total_loss / max(updates, 1)


# Compatibility alias for existing configs and imports.
OneStepSingleCriticPPOAgent = TrajectorySingleCriticPPOAgent
