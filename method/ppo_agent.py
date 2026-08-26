"""
ppo_agent.py - 基于 PyTorch 的 PPO 代理实现 (改进版)

改进点:
- Entropy Bonus (鼓励探索，防止策略过早收敛)
- State-dependent variance (状态相关的方差网络)
- Mini-batch 更新 (减少内存占用，提高收敛稳定性)
- Cosine annealing LR scheduler
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# ==================== 策略网络定义 ====================
class ActorCritic(nn.Module):
    """
    Actor-Critic 网络结构 (改进版)
    Actor: 输出动作均值和状态相关的方差
    Critic: 输出状态价值
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # 共享特征提取器
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor 网络 (策略网络) - 输出动作均值
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围 [-1, 1]，后续会缩放
        )
        
        # State-dependent log_std 网络
        # 替代固定参数 action_log_std，使方差随状态变化
        self.actor_log_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        
        # Critic 网络 (价值网络)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化 log_std 网络最后一层偏置为 -1 (初始 std≈0.37)
        nn.init.constant_(self.actor_log_std[-1].bias, -1.0)

    def forward(self, state: torch.Tensor):
        """前向传播，返回动作均值、方差和状态价值"""
        # 共享特征
        shared_features = self.shared(state)
        
        # 动作均值
        action_mean = self.actor_mean(shared_features) * 3.0  # 缩放到 [-3, 3]
        
        # 状态相关的动作 log_std (限制范围防止爆炸)
        action_log_std = self.actor_log_std(shared_features)
        action_log_std = torch.clamp(action_log_std, min=-5.0, max=2.0)
        action_std = torch.exp(action_log_std)
        
        # 状态价值
        state_value = self.critic(shared_features)
        
        return action_mean, action_std, state_value

# ==================== 经验回放缓冲区 ====================
class RolloutBuffer:
    """
    经验回放缓冲区
    功能：存储 PPO 训练所需的 (状态, 动作, 奖励, 对数概率) 数据
    """
    def __init__(self):
        self.states = []      # 状态列表 (numpy 数组)
        self.actions = []     # 动作列表 (numpy 数组)
        self.rewards = []     # 奖励列表 (标量)
        self.log_probs = []   # 动作对数概率列表 (标量)
        self.values = []      # 状态价值列表 (标量)
        self.dones = []       # 终止标志列表 (布尔值)
        self.entropies = []   # 策略熵列表 (用于 entropy bonus)

    def store(self, state: np.ndarray, action: np.ndarray, reward: float, 
              log_prob: float, value: float, done: bool, entropy: float = 0.0):
        """存储单步经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.entropies.append(entropy)

    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.entropies = []

    def get(self):
        """获取所有经验并转换为 Tensor"""
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(self.rewards)),
            torch.FloatTensor(np.array(self.log_probs)).unsqueeze(-1),
            torch.FloatTensor(np.array(self.values)).unsqueeze(-1),
            torch.FloatTensor(np.array(self.dones)),
            torch.FloatTensor(np.array(self.entropies)).unsqueeze(-1)
        )

    def __len__(self):
        return len(self.states)

# ==================== PPO 代理主体 ====================
class PPOAgent:
    """
    PPO 代理主体 (改进版)
    功能：使用 PPO 算法优化 VAE 的隐向量
    
    改进点:
    - Entropy Bonus: 损失中加入熵项，鼓励探索
    - Mini-batch 更新: 将经验分小批量更新，提高收敛稳定性
    - Cosine annealing LR: 余弦退火学习率调度
    """
    def __init__(self, state_dim: int, action_dim: int, 
                 lr: float = 3e-4, gamma: float = 0.99, 
                 gae_lambda: float = 0.95, 
                 ppo_clip: float = 0.2, 
                 ppo_epochs: int = 10,
                 entropy_coef: float = 0.01,
                 mini_batch_size: int = 32,
                 max_grad_norm: float = 1.0):
        """
        初始化 PPO 代理
        参数:
            state_dim: 状态维度 (隐向量维度)
            action_dim: 动作维度 (隐向量维度)
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE 的 lambda 参数
            ppo_clip: PPO 的 clip 范围
            ppo_epochs: 每次更新的 PPO 迭代次数
            entropy_coef: 熵系数 (Entropy Bonus 权重)
            mini_batch_size: Mini-batch 大小
            max_grad_norm: 梯度裁剪阈值
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        
        # 初始化网络和缓冲区
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = None  # 在训练过程中通过 init_scheduler 设置
        self.buffer = RolloutBuffer()
        
        # 统计
        self._total_updates = 0

    def init_scheduler(self, total_epochs: int):
        """初始化余弦退火学习率调度器"""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=1e-5
        )

    def step_scheduler(self):
        """学习率调度器步进"""
        if self.scheduler is not None:
            self.scheduler.step()

    def select_action(self, state: np.ndarray) -> tuple:
        """
        选择动作
        参数:
            state: 当前状态 (隐向量)
        返回:
            action: 选择的动作 (新的隐向量)
            log_prob: 动作的对数概率
            value: 状态价值
            entropy: 策略熵
        """
        with torch.no_grad():
            # 将 numpy 数组转换为 tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 前向传播
            action_mean, action_std, state_value = self.policy(state_tensor)
            
            # 构建多维正态分布
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            entropy = dist.entropy().sum(-1, keepdim=True)  # 总熵
            
            # 转换为 numpy 数组
            action = action.squeeze(0).numpy()
            log_prob = log_prob.squeeze(0).item()
            value = state_value.squeeze(0).item()
            entropy_val = entropy.squeeze(0).item()
            
            return action, log_prob, value, entropy_val

    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                         reward: float, log_prob: float, value: float, done: bool,
                         entropy: float = 0.0):
        """
        存储转移
        """
        self.buffer.store(state, action, reward, log_prob, value, done, entropy)

    def _compute_gae(self, rewards, values, dones):
        """
        计算广义优势估计 (GAE)
        """
        # 将tensor转换为numpy
        rewards = rewards.numpy()
        values = values.numpy()
        dones = dones.numpy()
        
        # 计算折扣回报
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        
        for t in reversed(range(len(rewards))):
            # 计算回报
            returns[t] = rewards[t] + self.gamma * prev_return * (1 - dones[t])
            # 计算优势
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            # GAE 计算
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * prev_advantage
            
            prev_return = returns[t]
            prev_value = values[t]
            prev_advantage = advantages[t]
        
        return torch.FloatTensor(advantages)

    def update(self):
        """
        更新策略网络 (Mini-batch 版本 + Entropy Bonus)
        功能：使用 PPO 算法更新策略
        """
        # 获取缓冲区数据
        states, actions, rewards, old_log_probs, values, dones, old_entropies = self.buffer.get()
        
        buffer_size = len(states)
        if buffer_size == 0:
            self.buffer.clear()
            return 0.0
        
        # 计算优势函数 (GAE)
        advantages = self._compute_gae(rewards, values, dones)
        
        # 计算回报 (用于价值函数训练)
        values = values.squeeze(-1)
        returns = advantages + values
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0.0
        n_batches = 0
        
        # PPO 迭代 (配合 Mini-batch)
        for _ in range(self.ppo_epochs):
            # 随机打乱数据索引
            indices = torch.randperm(buffer_size)
            
            # Mini-batch 更新
            for start in range(0, buffer_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, buffer_size)
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # 计算新的动作概率和状态价值
                action_means, action_stds, state_values = self.policy(mb_states)
                dist = torch.distributions.Normal(action_means, action_stds)
                action_log_probs = dist.log_prob(mb_actions).sum(-1, keepdim=True)
                
                # 计算策略熵
                entropy = dist.entropy().sum(-1, keepdim=True).mean()
                
                # 计算比率
                ratios = torch.exp(action_log_probs - mb_old_log_probs.detach())
                
                # 计算 PPO 损失
                surr1 = ratios * mb_advantages.detach()
                surr2 = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip) * mb_advantages.detach()
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                critic_loss = nn.MSELoss()(state_values, mb_returns.unsqueeze(-1))
                
                # Entropy Bonus (鼓励探索)
                entropy_loss = -self.entropy_coef * entropy
                
                # 总损失 = actor + critic + entropy_bonus
                loss = actor_loss + 0.5 * critic_loss + entropy_loss
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
        
        self._total_updates += 1
        
        # 清空缓冲区
        self.buffer.clear()
        
        avg_loss = total_loss / max(1, n_batches)
        return avg_loss

    def get_lr(self) -> float:
        """获取当前学习率"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def save_model(self, path: str):
        """
        保存模型
        参数:
            path: 保存路径
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_updates': self._total_updates,
        }, path)

    def load_model(self, path: str):
        """
        加载模型
        参数:
            path: 模型路径
        """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._total_updates = checkpoint.get('total_updates', 0)
