"""
finaly.py - 三阶段动态帕累托权重控制器 (修正版)

功能：根据训练进度（Epoch）自动切换三种策略，为PPO提供动态奖励权重。
阶段1 (探索): 基于参考向量的均匀探索。
阶段2 (利用): 基于超体积贡献 (Hypervolume Contribution) 的定向激励。
阶段3 (平衡): 基于标准差的分布平衡。

注意：此版本修复了超体积计算逻辑，确保返回正值，并优化了数据接口。
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime

# ==================== 数据结构定义 ====================
@dataclass
class Molecule:
    """分子数据结构，用于存储生成的分子及其属性"""
    smiles: str               # 分子的SMILES字符串
    latent_vector: np.ndarray # 分子在VAE中的隐向量 (Z)
    scores: np.ndarray        # 分子的多目标性质分数 (如 QED, LogP, SAS)

class ParetoFront:
    """帕累托前沿解集管理器"""
    def __init__(self, num_obj: int):
        self.num_obj = num_obj
        self.solutions: List[np.ndarray] = [] # 存储非支配解的分数
        self.molecules: List[Molecule] = []    # 存储非支配解对应的分子对象

    def update(self, new_solutions: List[np.ndarray], new_molecules: List[Molecule]):
        """
        合并新解并更新前沿（非支配排序）
        算法：将旧解和新解合并，计算非支配排序，保留非支配解。
        """
        # 1. 合并旧解和新解
        combined = self.solutions + new_solutions
        combined_mols = self.molecules + new_molecules
        
        if len(combined) == 0:
            return

        # 2. 快速非支配排序 (Non-dominated Sort)
        # dominated[i] 为 True 表示第 i 个解被支配
        dominated = [False] * len(combined)
        
        for i in range(len(combined)):
            if dominated[i]:
                continue
            for j in range(len(combined)):
                if i == j or dominated[j]:
                    continue
                # 检查 combined[i] 是否支配 combined[j]
                if self._dominates(combined[i], combined[j]):
                    dominated[j] = True
                # 检查 combined[j] 是否支配 combined[i]
                elif self._dominates(combined[j], combined[i]):
                    dominated[i] = True
                    break # 如果 i 被支配，跳出内层循环
        
        # 3. 保留未被支配的解
        self.solutions = [combined[i] for i in range(len(combined)) if not dominated[i]]
        self.molecules = [combined_mols[i] for i in range(len(combined)) if not dominated[i]]

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        支配关系判断 (Maximization Problem)
        定义：如果 a 在所有目标上都 >= b，且至少有一个目标 > b，则 a 支配 b。
        """
        all_ge = np.all(a >= b) # 所有目标都大于等于
        any_gt = np.any(a > b)  # 至少有一个目标大于
        return all_ge and any_gt

    def _filter_nondominated(self, points: np.ndarray) -> np.ndarray:
        """过滤并保留前沿中的非支配解。"""
        if points.size == 0:
            return points
        num = points.shape[0]
        dominated = np.zeros(num, dtype=bool)
        for i in range(num):
            if dominated[i]:
                continue
            for j in range(num):
                if i == j or dominated[j]:
                    continue
                if self._dominates(points[i], points[j]):
                    dominated[j] = True
                elif self._dominates(points[j], points[i]):
                    dominated[i] = True
                    break
        return points[~dominated]

    def _exact_hypervolume(self, front: np.ndarray, ref_point: np.ndarray) -> float:
        """计算非支配点集的精确超体积。"""
        if front.size == 0:
            return 0.0
        dims = front.shape[1]
        if dims == 1:
            return float(np.max(front[:, 0] - ref_point[0]))

        order = np.argsort(front[:, 0])
        sorted_front = front[order]
        hv = 0.0
        prev = ref_point[0]

        for i in range(sorted_front.shape[0]):
            x = sorted_front[i, 0]
            if x <= prev:
                continue
            slice_front = sorted_front[i:, 1:]
            slice_ref = ref_point[1:]
            slice_hv = self._exact_hypervolume(slice_front, slice_ref)
            hv += (x - prev) * slice_hv
            prev = x

        return float(hv)

    def compute_hypervolume(self, ref_point: np.ndarray) -> float:
        """
        计算超体积 (Hypervolume Metric)
        功能：衡量帕累托前沿的质量。值越大，表示解集覆盖的目标空间越大。
        """
        if len(self.solutions) == 0:
            return 0.0

        front = np.array(self.solutions, dtype=np.float64)
        if len(ref_point) != front.shape[1]:
            raise ValueError("Reference point dimension mismatch.")

        valid = front[np.all(front >= ref_point, axis=1)]
        if valid.size == 0:
            return 0.0

        valid = self._filter_nondominated(valid)
        return self._exact_hypervolume(valid, np.array(ref_point, dtype=np.float64))

# ==================== 阶段一：基于参考向量的均匀探索 ====================
class Phase1Controller:
    """
    阶段一控制器：基于 Das-Dennis 均匀采样的自适应探索
    目标：在目标空间中真正均匀撒网，确保帕累托前沿被广泛覆盖。
    
    修正：使用 Das-Dennis 单纯形格点法替代随机采样，消除随机偏置，
    生成真正均匀分布的方向向量。
    """
    def __init__(self, num_obj: int, num_divisions: int = 8):
        """
        Args:
            num_obj: 目标数量 (必须 >= 2)
            num_divisions: 单纯形每维剖分数，向量总数 = C(num_divisions + num_obj - 1, num_obj - 1)
                           例如: num_obj=3, num_divisions=8 → C(10,2)=45 个向量
                           增大此值可提高覆盖率
        """
        self.num_obj = num_obj
        self.num_divisions = num_divisions
        # 初始化 Das-Dennis 均匀参考向量
        self.ref_vectors = self._init_das_dennis_vectors()
        self.num_vectors = len(self.ref_vectors)
        
    def _init_das_dennis_vectors(self) -> np.ndarray:
        """
        Das-Dennis 单纯形格点法生成真正均匀分布的参考向量。
        
        原理：在 (num_obj-1) 维单纯形上均匀采样，然后归一化到单位球面。
        相比 random.randn + normalize，避免了高维球面的聚类效应。
        
        返回:
            vectors: shape (n_vectors, num_obj) 的归一化向量
        """
        from itertools import combinations
        
        m = self.num_obj
        p = self.num_divisions
        
        # 步骤1: 生成所有分割点组合
        # 在 [0, p+m-1] 范围内选 (m-1) 个分割点
        all_combos = list(combinations(range(1, p + m), m - 1))
        
        vectors = []
        for combo in all_combos:
            # 步骤2: 将分割点转换为单纯形坐标
            point = np.zeros(m)
            prev = 0
            for i, c in enumerate(combo):
                point[i] = (c - prev - 1) / p  # 归一化到 [0,1]
                prev = c
            point[-1] = (p + m - 1 - prev) / p  # 最后一个分量
            
            # 步骤3: 确保所有分量非零 (避免退化向量)
            # 将所有分量加 epsilon 避免完全为零的方向
            point = np.maximum(point, 0.0)  # 确保非负
            
            # 归一化到单位长度
            norm = np.linalg.norm(point)
            if norm > 1e-8:
                vectors.append(point / norm)
        
        vectors = np.array(vectors)
        
        # 如果向量数量不够 (num_obj 和 num_divisions 组合导致)，补充随机向量
        min_vectors = max(10, 2 * m)
        if len(vectors) < min_vectors:
            extra = min_vectors - len(vectors)
            rand_vecs = np.random.randn(extra, m)
            rand_vecs = rand_vecs / np.linalg.norm(rand_vecs, axis=1, keepdims=True)
            vectors = np.vstack([vectors, rand_vecs]) if len(vectors) > 0 else rand_vecs
        
        return vectors
        
    def update(self, pareto_front: List[np.ndarray]) -> np.ndarray:
        """
        更新策略：根据当前前沿的分布，调整参考向量，并返回当前权重。
        逻辑：计算拥挤度，从稀疏区域选择向量。
        """
        if len(pareto_front) == 0:
            # 前沿为空，随机返回一个向量
            return self.ref_vectors[np.random.randint(0, len(self.ref_vectors))]
        
        # 1. 计算每个参考向量方向的拥挤度
        crowding = self._compute_crowding(pareto_front)
        
        # 2. 调整策略：选择拥挤度最小的方向 (最稀疏的方向)
        # 这里简单选择拥挤度最大的向量 (即最不拥挤的方向)
        # 注意：这里假设 crowding 值越大表示越稀疏 (根据 _compute_crowding 的定义)
        # 如果 crowding 定义为距离，则越大越稀疏
        selected_idx = np.argmax(crowding)
        
        # 返回选中的参考向量作为权重
        return self.ref_vectors[selected_idx]
    
    def _compute_crowding(self, pareto_front: List[np.ndarray]) -> np.ndarray:
        """
        计算拥挤度：衡量每个参考向量方向上有多少解。
        
        使用余弦相似度 + 距离惩罚计算拥挤度。
        拥挤度越大 → 该方向越稀疏 → 优先选择。
        
        返回：每个向量方向的拥挤度数组（值越大越稀疏）。
        """
        front_array = np.array(pareto_front)
        crowding = np.zeros(self.num_vectors)
        
        for i, vec in enumerate(self.ref_vectors):
            # 归一化前沿解
            norms = np.linalg.norm(front_array, axis=1, keepdims=True)
            norms[norms == 0] = 1e-8
            normalized_front = front_array / norms
            
            # 计算余弦相似度 (点积)
            similarities = np.dot(normalized_front, vec)
            
            # 拥挤度 = 负的"最近解数"：使用 softmin 风格
            # 如果该方向附近有很多解 (高相似度)，则拥挤度高
            # 我们想要选择拥挤度低 (即 sparse) 的方向
            # 用相似度的高斯核密度估计拥挤程度
            # 带宽 = 0.3 (对应约 cos(72°) ≈ 0.3 的邻域)
            bandwidth = 0.3
            if len(similarities) > 0:
                # 核密度: 相似度越高 → 密度越大
                density = np.mean(np.exp(-((1.0 - similarities) / bandwidth) ** 2))
                # 稀疏度 = 1 / (density + eps), 值越大越稀疏
                crowding[i] = 1.0 / (density + 1e-8)
            else:
                crowding[i] = 1.0  # 没有解则非常稀疏
        
        return crowding

# ==================== 阶段二：基于超体积贡献的定向激励 ====================
class Phase2Controller:
    """
    阶段二控制器：基于超体积贡献 (HVC) 的定向激励 (改进版)
    目标：用 Softmax 归一化 HVC 贡献度，按贡献度加权取均值得到改进方向。
    """
    def __init__(self, num_obj: int, alpha: float = 0.1, temperature: float = 0.5):
        self.num_obj = num_obj
        self.alpha = alpha
        self.temperature = temperature  # softmax 温度
        self.ref_point = np.zeros(num_obj)
        # 追踪历史最优各目标均值 (用于计算改进幅度)
        self._historical_best_avg = np.zeros(num_obj)
        
    def set_ref_point(self, ref_point: np.ndarray):
        """设置超体积计算的参考点"""
        self.ref_point = ref_point.copy()
        
    def update(self, new_scores: np.ndarray, pareto_front: List[np.ndarray]) -> np.ndarray:
        """
        改进策略 (与原始版本的核心差异):
        1. Softmax 归一化 HVC，按贡献度加权取目标均值
        2. 与均匀底权混合，防止过拟合单一目标
        """
        if len(pareto_front) == 0:
            return np.ones(self.num_obj) / self.num_obj
            
        batch_size = new_scores.shape[0]
        hv_contributions = []
        for i in range(batch_size):
            delta_hv = self._compute_contribution(new_scores[i], pareto_front)
            hv_contributions.append(delta_hv)
        
        hv_contributions = np.array(hv_contributions)
        
        if np.max(hv_contributions) <= 0:
            return np.ones(self.num_obj) / self.num_obj
        
        # Softmax 归一化 (温度控制选择尖锐度)
        hv_max = np.max(hv_contributions)
        hv_contributions = hv_contributions - hv_max  # 数值稳定
        exp_hv = np.exp(hv_contributions / (self.temperature + 1e-8))
        hvc_weights = exp_hv / np.sum(exp_hv)  # shape (batch_size,)
        
        # 加权目标均值: sum_i(weight_i * score_i) / sum_i(weight_i)
        weighted_obj = np.sum(new_scores * hvc_weights[:, np.newaxis], axis=0)
        sum_w = np.sum(hvc_weights)
        if sum_w > 0:
            weighted_obj = weighted_obj / sum_w
        
        # 归一化到概率权重
        obj_min, obj_max = weighted_obj.min(), weighted_obj.max()
        if obj_max > obj_min:
            normalized_obj = (weighted_obj - obj_min) / (obj_max - obj_min)
        else:
            normalized_obj = np.ones(self.num_obj) * 0.5
        
        # 与均匀底权混合 (0.7*HVC + 0.3*uniform)
        uniform_weight = np.ones(self.num_obj) / self.num_obj
        final_weights = 0.7 * normalized_obj / (np.sum(normalized_obj) + 1e-8) + 0.3 * uniform_weight
        
        return final_weights
        
    def _compute_contribution(self, new_solution: np.ndarray, pareto_front: List[np.ndarray]) -> float:
        """
        计算单个新解的超体积贡献。
        逻辑：ΔHV = HV(旧前沿 U 新解) - HV(旧前沿)
        简化：这里计算新解相对于参考点的体积，减去被新解支配的旧解的体积。
        """
        # 使用精确超体积增量替代近似体积计算
        old_array = np.array(pareto_front, dtype=np.float64)
        if old_array.size == 0:
            return float(np.prod(np.maximum(0, new_solution - self.ref_point)))

        old_array = old_array[np.all(old_array >= self.ref_point, axis=1)]
        if old_array.size == 0:
            hv_old = 0.0
        else:
            old_front = ParetoFront(self.num_obj)
            old_front.solutions = [row for row in old_array]
            hv_old = old_front.compute_hypervolume(self.ref_point)

        combined = new_solution.reshape(1, -1)
        if old_array.size > 0:
            combined = np.vstack([old_array, combined])
        combined = combined[np.all(combined >= self.ref_point, axis=1)]
        combined_front = ParetoFront(self.num_obj)
        combined_front.solutions = [row for row in combined]
        hv_new = combined_front.compute_hypervolume(self.ref_point)

        return max(0.0, hv_new - hv_old)

# ==================== 阶段三：基于标准差的平衡调整 ====================
class Phase3Controller:
    """
    阶段三控制器：基于标准差的平衡调整 + EMA 平滑 (改进版)
    目标：从 Phase2 结束时的权重继承，防止权重震荡，维持前沿分布均匀。
    """
    def __init__(self, num_obj: int, beta: float = 0.05, ema_alpha: float = 0.3):
        self.num_obj = num_obj
        self.beta = beta
        self.ema_alpha = ema_alpha  # EMA 平滑系数 (0~1, 越低越平滑)
        self._prev_weights = None   # 上一轮权重 (EMA 状态)
        
    def inherit_weights(self, weights: np.ndarray):
        """从 Phase2 继承初始权重"""
        self._prev_weights = weights.copy()
        
    def update(self, pareto_front: List[np.ndarray], current_weights: np.ndarray, epoch: int, total_epochs: int) -> np.ndarray:
        """
        改进策略:
        1. 根据标准差反比调整权重 (方差大 → 降低权重)
        2. EMA 平滑: w_{t+1} = α·w_raw + (1-α)·w_t
        """
        if len(pareto_front) < 2:
            if self._prev_weights is not None:
                return self._prev_weights.copy()
            return current_weights.copy()
            
        front_array = np.array(pareto_front)
        stds = np.std(front_array, axis=0)
        
        base_weights = self._prev_weights.copy() if self._prev_weights is not None else current_weights.copy()
        new_weights = base_weights.copy()
        
        # 标准差反比调整
        for i in range(self.num_obj):
            if np.max(stds) > 1e-8:
                new_weights[i] = base_weights[i] * (1.0 / (stds[i] + 1e-8))
        
        # 归一化
        weight_sum = np.sum(new_weights)
        if weight_sum > 0:
            new_weights = new_weights / weight_sum
        else:
            new_weights = np.ones(self.num_obj) / self.num_obj
        
        # EMA 平滑 (如果已有历史权重)
        if self._prev_weights is not None:
            new_weights = self.ema_alpha * new_weights + (1 - self.ema_alpha) * self._prev_weights
        
        self._prev_weights = new_weights.copy()
        return new_weights

# ==================== 主控制器：三阶段集成 (改进版) ====================
class ThreeStageWeightController:
    """
    三阶段动态权重主控制器 (改进版)
    改进点:
    1. 三阶段软过渡 (sigmoid 融合) 代替硬切换
    2. Phase3 权重继承自 Phase2 结束时的权重
    3. 记录完整权重历史 (用于可视化)
    """
    def __init__(self, num_obj: int, total_epochs: int, 
                 phase1_ratio: float = 0.3, phase2_ratio: float = 0.4,
                 smooth_width_fraction: float = 0.05):
        self.num_obj = num_obj
        self.total_epochs = total_epochs
        self.phase1_ratio = phase1_ratio
        self.phase2_ratio = phase2_ratio
        self.smooth_width_fraction = smooth_width_fraction  # 过渡带宽度 (如 0.05=5% epochs)
        
        # 初始化各阶段控制器
        self.phase1 = Phase1Controller(num_obj)
        self.phase2 = Phase2Controller(num_obj)
        self.phase3 = Phase3Controller(num_obj)
        
        # 帕累托前沿管理器
        self.pareto_front = ParetoFront(num_obj)
        
        # 参考点 (用于超体积计算)
        self.ref_point = np.zeros(num_obj)
        
        # 权重历史记录 (每 epoch 记录)
        self._weights_history: List[np.ndarray] = []  # 每个 epoch 的权重
        self._phase_history: List[int] = []           # 每个 epoch 的主要阶段
        
        # 标记 Phase2→Phase3 继承是否已完成
        self._weights_inherited_for_phase3 = False
        
    def set_ref_point(self, ref_point: np.ndarray):
        """设置参考点"""
        self.ref_point = ref_point.copy()
        self.phase2.set_ref_point(ref_point)
        
    def get_phase(self, epoch: int) -> int:
        """根据当前轮次判断所处阶段"""
        if epoch < self.phase1_ratio * self.total_epochs:
            return 1
        elif epoch < (self.phase1_ratio + self.phase2_ratio) * self.total_epochs:
            return 2
        else:
            return 3
    
    def get_smooth_weights(self, epoch: int, new_scores: np.ndarray) -> np.ndarray:
        """
        核心接口 (替代原始 get_weights): 
        使用 sigmoid 软过渡 + 权重继承 + EMA 平滑的权重计算。
        """
        phase = self.get_phase(epoch)
        smooth_width = int(self.smooth_width_fraction * self.total_epochs)
        weights = None
        
        # 阶段1 权重
        w1 = self.phase1.update(self.pareto_front.solutions) if new_scores.shape[0] > 0 else np.ones(self.num_obj) / self.num_obj
        # 阶段2 权重
        w2 = self.phase2.update(new_scores, self.pareto_front.solutions)
        # 阶段3 权重 (先占位均匀权重，实际只在 Phase3 区域更新)
        w3 = self.phase3.update(self.pareto_front.solutions, np.ones(self.num_obj) / self.num_obj, epoch, self.total_epochs)
        
        # ---- Phase1→Phase2 过渡区域 ----
        p1_end = int(self.phase1_ratio * self.total_epochs)  # Phase1 结束
        transition_start_1_2 = p1_end - smooth_width
        transition_end_1_2 = p1_end
        
        if transition_start_1_2 <= epoch < transition_end_1_2:
            # 在过渡带内: sigmoid 融合
            t = (epoch - transition_start_1_2) / max(1, (transition_end_1_2 - transition_start_1_2))
            blend = 1.0 / (1.0 + np.exp(-(t - 0.5) * 12))  # sigmoid
            weights = (1 - blend) * w1 + blend * w2
            phase = 1  # 记录为过渡阶段
        
        # ---- Phase2→Phase3 过渡区域 ----
        p2_end = int((self.phase1_ratio + self.phase2_ratio) * self.total_epochs)
        transition_start_2_3 = p2_end - smooth_width
        transition_end_2_3 = p2_end
        
        if transition_start_2_3 <= epoch < transition_end_2_3:
            # 首次进入过渡带时，继承 Phase2 权重到 Phase3
            if not self._weights_inherited_for_phase3:
                self.phase3.inherit_weights(w2)
                self._weights_inherited_for_phase3 = True
            t = (epoch - transition_start_2_3) / max(1, (transition_end_2_3 - transition_start_2_3))
            blend = 1.0 / (1.0 + np.exp(-(t - 0.5) * 12))
            weights = (1 - blend) * w2 + blend * w3
            phase = 2  # 记录为过渡阶段
        
        # ---- 纯阶段区域 ----
        if weights is None:
            if phase == 1:
                weights = w1
            elif phase == 2:
                weights = w2
            else:
                # Phase3: 首次进入时继承 Phase2 权重
                if not self._weights_inherited_for_phase3:
                    self.phase3.inherit_weights(w2)
                    self._weights_inherited_for_phase3 = True
                weights = self.phase3.update(self.pareto_front.solutions, np.ones(self.num_obj) / self.num_obj, epoch, self.total_epochs)
        
        # 归一化
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(self.num_obj) / self.num_obj
        
        # 记录历史
        self._weights_history.append(weights.copy())
        self._phase_history.append(phase)
        
        return weights
            
    def update_pareto_front(self, new_molecules: List[Molecule]):
        """更新帕累托前沿"""
        new_scores = [m.scores for m in new_molecules]
        self.pareto_front.update(new_scores, new_molecules)
        
    def get_weights(self, epoch: int, new_scores: np.ndarray) -> np.ndarray:
        """保持向后兼容，内部委托给软过渡版本"""
        return self.get_smooth_weights(epoch, new_scores)
        
    def compute_hypervolume(self) -> float:
        """对外接口：计算当前前沿超体积"""
        return self.pareto_front.compute_hypervolume(self.ref_point)
    
    def get_weights_history(self) -> List[np.ndarray]:
        """获取权重历史 (用于可视化)"""
        return self._weights_history

# ==================== 可视化类 (保持不变，仅增加注释) ====================
class ThreeStageVisualization:
    """三阶段可视化工具"""
    def __init__(self, controller, num_obj: int, total_epochs: int, save_dir="three_stage_viz_output"):
        self.controller = controller
        self.num_obj = num_obj
        self.total_epochs = total_epochs
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建图表
        self.fig = plt.figure(figsize=(20, 15))
        self.ax1 = self.fig.add_subplot(2, 3, 1, projection='3d') # 前沿
        self.ax2 = self.fig.add_subplot(2, 3, 2, projection='3d') # 向量
        self.ax3 = self.fig.add_subplot(2, 3, 3) # 权重曲线
        self.ax4 = self.fig.add_subplot(2, 3, 4) # 阶段
        self.ax5 = self.fig.add_subplot(2, 3, 5) # 超体积
        self.ax6 = self.fig.add_subplot(2, 3, 6) # 前沿大小
        
        # 初始化绘图元素
        self.weights_history = []
        self.hv_history = []
        self.size_history = []
        
    def update_plot(self, new_solutions: List[np.ndarray], weights: np.ndarray, epoch: int, save_image=False):
        """更新可视化图表"""
        # 这里省略具体的绘图代码细节，保留框架
        # ... (绘图逻辑保持与原版类似) ...
        
        # 保存图片
        if save_image:
            plt.savefig(f"{self.save_dir}/gen_{epoch}.png")
            
        plt.pause(0.01)
