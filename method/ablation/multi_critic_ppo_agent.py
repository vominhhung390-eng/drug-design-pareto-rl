"""
multi_critic_ppo_agent.py — 多 Critic PPO 代理

核心改进:
  1. 偏好向量条件化: Actor 输入 = concat(z, ω)，策略学习"给定偏好→生成对应分子"
  2. 多目标 Critic: 3 个独立的 V_i(s, ω) 分别预测 EGFR/VEGFR2/QED 的值函数
  3. 标量化 Advantage: A_combined = Σ_i ω_i · A_i，但每个 Critic 独立训练
  4. 中 Critic 一致性: 各自的值函数不受动态权重污染，GAE 估计更稳定

用法:
  from multi_critic_ppo_agent import MultiCriticPPOAgent
  agent = MultiCriticPPOAgent(state_dim=128, action_dim=128, num_obj=3)
  
  # 采样偏好向量
  omega = np.random.dirichlet([0.5, 0.5, 0.5])
  
  # 选择动作 (actor 输入 = concat(z, omega))
  action, log_prob, values, entropy = agent.select_action(state, omega)
  
  # values 是 shape (3,) 的多目标值函数
  # 存储多维奖励和偏好
  agent.store_transition(state, action, scores, log_prob, values, done, entropy, omega)
  
  # 更新 (内部自动标量化 advantage)
  loss = agent.update()
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ==================== 多 Critic Actor-Critic 网络 ====================
class MultiCriticActorCritic(nn.Module):
    """
    偏好条件化的 Actor-Critic 网络 (多 Critic 版)
    
    输入: concat(z, ω) → state_dim + num_obj 维
    输出:
      - action_mean: (action_dim,) 动作均值
      - action_std: (action_dim,) 动作标准差
      - critic_values: (num_obj,) 多个目标的值函数
    """
    def __init__(self, state_dim: int, action_dim: int, num_obj: int = 3,
                 hidden_dim: int = 256):
        super(MultiCriticActorCritic, self).__init__()
        self.num_obj = num_obj
        input_dim = state_dim + num_obj  # 拼接偏好向量

        # 共享特征提取器
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor 网络 (策略网络) - 输出动作均值
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # 输出范围 [-1, 1]，后续会缩放
        )

        # State-dependent log_std 网络
        self.actor_log_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # === 多 Critic 头: 每个目标独立的值函数 ===
        self.critic_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_obj)
        ])

        # 初始化
        nn.init.constant_(self.actor_log_std[-1].bias, -1.0)  # 初始 std ≈ 0.37

    def forward(self, state: torch.Tensor, preference: torch.Tensor):
        """
        前向传播
        
        Args:
            state: (batch, state_dim) 隐向量
            preference: (batch, num_obj) 偏好向量 ω
        
        Returns:
            action_mean: (batch, action_dim)
            action_std: (batch, action_dim)
            critic_values: List[(batch, 1)] * num_obj
        """
        # 拼接隐向量和偏好向量
        combined = torch.cat([state, preference], dim=-1)

        # 共享特征
        shared_features = self.shared(combined)

        # 动作均值 (缩放到 [-3, 3])
        action_mean = self.actor_mean(shared_features) * 3.0

        # 动作 log_std (裁剪防爆炸)
        action_log_std = self.actor_log_std(shared_features)
        action_log_std = torch.clamp(action_log_std, min=-5.0, max=2.0)
        action_std = torch.exp(action_log_std)

        # 多目标值函数
        critic_values = [head(shared_features) for head in self.critic_heads]

        return action_mean, action_std, critic_values


# ==================== 多 Critic 经验回放缓冲区 ====================
class MultiCriticRolloutBuffer:
    """
    存储多目标经验: 多维奖励 + 多维价值 + 偏好向量
    """
    def __init__(self):
        self.states = []       # 隐向量 (N, state_dim)
        self.actions = []      # 动作 (N, action_dim)
        self.rewards = []      # 多维奖励 (N, num_obj)
        self.log_probs = []    # 对数概率 (标量/批)
        self.values = []       # 多维价值 (N, num_obj)
        self.dones = []        # 终止标志
        self.entropies = []    # 策略熵
        self.preferences = []  # 偏好向量 (N, num_obj)
        # Scalar Pareto-aware shaping used by the actor only. Keeping this
        # separate preserves the semantic meaning of the objective critics.
        self.auxiliary_rewards = []

    def store(self, state: np.ndarray, action: np.ndarray,
              rewards: np.ndarray,   # shape (num_obj,)
              log_prob: float,
              values: np.ndarray,    # shape (num_obj,)
              done: bool,
              entropy: float = 0.0,
              preference: np.ndarray = None,
              auxiliary_reward: float = 0.0):
        """存储单步多目标经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(rewards)
        self.log_probs.append(log_prob)
        self.values.append(values)
        self.dones.append(done)
        self.entropies.append(entropy)
        self.preferences.append(preference if preference is not None else
                                np.ones(len(rewards)) / len(rewards))
        self.auxiliary_rewards.append(float(auxiliary_reward))

    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.entropies = []
        self.preferences = []
        self.auxiliary_rewards = []

    def get(self):
        """获取所有经验并转换为 Tensor"""
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(self.rewards)),          # (N, num_obj)
            torch.FloatTensor(np.array(self.log_probs)).unsqueeze(-1),
            torch.FloatTensor(np.array(self.values)),           # (N, num_obj)
            torch.FloatTensor(np.array(self.dones)),
            torch.FloatTensor(np.array(self.entropies)).unsqueeze(-1),
            torch.FloatTensor(np.array(self.preferences)),      # (N, num_obj)
        )

    def __len__(self):
        return len(self.states)


# ==================== 多 Critic PPO 代理 ====================
class MultiCriticPPOAgent:
    """
    多 Critic PPO 代理
    
    核心机制:
      1. Actor 输入 = concat(z, ω): 偏好条件化策略
      2. 3 个独立 Critic: 每个预测对应目标的 V_i
      3. 标量化 Advantage: A_comb = Σ_i ω_i · A_i
      4. 独立 Critic 训练: 每个 Critic 用各自目标的回报训练
    """
    def __init__(self, state_dim: int, action_dim: int, num_obj: int = 3,
                 lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 ppo_clip: float = 0.2,
                 ppo_epochs: int = 10,
                 entropy_coef: float = 0.01,
                 mini_batch_size: int = 32,
                 max_grad_norm: float = 1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_obj = num_obj
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm

        # 多 Critic 策略网络
        self.policy = MultiCriticActorCritic(state_dim, action_dim, num_obj)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = None
        self.buffer = MultiCriticRolloutBuffer()

        self._total_updates = 0

    def init_scheduler(self, total_epochs: int):
        """初始化余弦退火学习率调度器"""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=1e-5
        )

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def select_action(self, state: np.ndarray, preference: np.ndarray = None) -> tuple:
        """
        选择动作 (偏好条件化)
        
        Args:
            state: (state_dim,) 隐向量
            preference: (num_obj,) 偏好向量，默认均匀
        
        Returns:
            action: (action_dim,) 动作
            log_prob: 对数概率
            values: (num_obj,) 多目标值函数
            entropy: 策略熵
        """
        if preference is None:
            preference = np.ones(self.num_obj) / self.num_obj

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)       # (1, state_dim)
            pref_t = torch.FloatTensor(preference).unsqueeze(0)    # (1, num_obj)

            action_mean, action_std, critic_values = self.policy(state_t, pref_t)

            # 采样动作
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            entropy = dist.entropy().sum(-1, keepdim=True)

            # 合并多 Critic 值 → (num_obj,)
            values = torch.cat([v for v in critic_values], dim=-1)  # (1, num_obj)

            action = action.squeeze(0).numpy()
            log_prob = log_prob.squeeze(0).item()
            values = values.squeeze(0).numpy()                      # (num_obj,)
            entropy_val = entropy.squeeze(0).item()

            return action, log_prob, values, entropy_val

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         reward: float, log_prob: float, value: float,
                         done: bool, entropy: float = 0.0):
        """
        兼容原始 PPOAgent 的单标量接口 (main_pipeline 调用此方法)
        
        内部会覆盖为多目标语义: 调用者应先调用 store_multi 或通过
        run_episode 适配层处理。
        """
        raise NotImplementedError(
            "MultiCriticPPOAgent 需要多维奖励和价值，请使用 store_transition_multi() 方法。"
            "在 main_pipeline 中使用时需通过适配层将标量接口转换为多目标接口。"
        )

    def store_transition_multi(self, state: np.ndarray, action: np.ndarray,
                               rewards: np.ndarray,    # (num_obj,)
                               log_prob: float,
                               values: np.ndarray,     # (num_obj,)
                               done: bool,
                               entropy: float = 0.0,
                               preference: np.ndarray = None,
                               auxiliary_reward: float = 0.0):
        """存储多目标经验"""
        self.buffer.store(state, action, rewards, log_prob, values, done,
                          entropy, preference, auxiliary_reward)

    def _compute_multi_gae(self, rewards, values, dones):
        """
        对每个目标独立计算 GAE advantage
        
        Args:
            rewards: (N, num_obj) 多维奖励
            values: (N, num_obj) 多维价值
            dones: (N,) 终止标志
        
        Returns:
            advantages: (N, num_obj) 每个目标的 GAE advantage
        """
        rewards_np = rewards.numpy()
        values_np = values.numpy()
        dones_np = dones.numpy()

        N, K = rewards_np.shape
        advantages = np.zeros_like(rewards_np)

        for k in range(K):
            prev_advantage = 0.0
            for t in reversed(range(N)):
                if t == N - 1:
                    next_value = 0.0 if dones_np[t] else values_np[t, k]
                else:
                    next_value = values_np[t + 1, k]

                delta = (rewards_np[t, k]
                         + self.gamma * next_value * (1 - dones_np[t])
                         - values_np[t, k])
                advantages[t, k] = (delta
                                    + self.gamma * self.gae_lambda
                                    * (1 - dones_np[t]) * prev_advantage)
                prev_advantage = advantages[t, k]

        return torch.FloatTensor(advantages)

    def update(self):
        """
        多 Critic PPO 更新
        
        流程:
          1. 对每个目标独立计算 GAE advantage → A_i
          2. 标量化: A_comb = Σ ω · A_i
          3. 对每个 Critic 独立计算 MSE loss (使用各自目标的 returns)
          4. Actor loss: 标准 PPO clip (使用 A_comb)
          5. Entropy bonus
        """
        (states, actions, rewards, old_log_probs,
         values, dones, old_entropies, preferences) = self.buffer.get()

        buffer_size = len(states)
        if buffer_size == 0:
            self.buffer.clear()
            return 0.0

        K = self.num_obj

        # 1. 对每个目标独立计算 GAE advantage
        advantages = self._compute_multi_gae(rewards, values, dones)  # (N, K)

        # 2. 标量化 advantage: A_comb = Σ ω · A_i
        advantages_combined = (advantages * preferences).sum(dim=-1)  # (N,)

        # 3. 每个目标的 returns = advantage_i + value_i
        returns = advantages + values  # (N, K)

        # 归一化标量化 advantage
        adv_mean = advantages_combined.mean()
        adv_std = advantages_combined.std()
        advantages_combined = ((advantages_combined - adv_mean)
                               / (adv_std + 1e-8))

        total_loss = 0.0
        n_batches = 0

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(buffer_size)

            for start in range(0, buffer_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, buffer_size)
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages_combined[mb_idx]
                mb_returns = returns[mb_idx]          # (mb, K)
                mb_prefs = preferences[mb_idx]        # (mb, K)

                # 前向传播 (条件化)
                action_means, action_stds, critic_values = self.policy(
                    mb_states, mb_prefs
                )

                # --- Actor Loss ---
                dist = torch.distributions.Normal(action_means, action_stds)
                action_log_probs = dist.log_prob(mb_actions).sum(-1, keepdim=True)

                ratios = torch.exp(action_log_probs - mb_old_log_probs.detach())
                surr1 = ratios * mb_advantages.detach().unsqueeze(-1)
                surr2 = (torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip)
                         * mb_advantages.detach().unsqueeze(-1))
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Multi-Critic Loss (每个 Critic 独立) ---
                critic_loss = 0.0
                for k in range(K):
                    # critic_values[k]: (mb, 1)
                    # mb_returns[:, k]: (mb,)
                    critic_loss += nn.MSELoss()(
                        critic_values[k].squeeze(-1),
                        mb_returns[:, k].detach()
                    )
                critic_loss = critic_loss / K  # 平均

                # --- Entropy Bonus ---
                entropy = dist.entropy().sum(-1, keepdim=True).mean()
                entropy_loss = -self.entropy_coef * entropy

                # --- 总损失 ---
                loss = actor_loss + 0.5 * critic_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), max_norm=self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        self._total_updates += 1
        self.buffer.clear()

        return total_loss / max(1, n_batches)

    def get_lr(self) -> float:
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def save_model(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_updates': self._total_updates,
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._total_updates = checkpoint.get('total_updates', 0)
