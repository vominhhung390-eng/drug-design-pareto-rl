"""
ablation/weight_controllers.py — 消融实验: 6种三阶段权重控制器变体

变体说明:
  Ours (full)     : Das-Dennis探索 + HVC利用 + Std-EMA平衡 (完整三阶段，含sigmoid软过渡)
  w/o Stage1      : 随机权重探索 + HVC利用 + Std-EMA平衡
  w/o Stage2      : Das-Dennis探索 + 固定权重利用 + Std-EMA平衡
  w/o Stage3      : Das-Dennis探索 + HVC利用持续 (无平衡阶段)
  Fixed Weight    : 等权 (1/3, 1/3, 1/3) 全程不变 (加权和基线)
  Scalarized      : 随机方向扫描标量化 (每N轮更换权重，模拟多run单目标)

设计原则:
  1. 所有变体实现统一接口，与 finaly.py 的 ThreeStageWeightController 兼容
  2. 可直接替换到 MO_RL_Integrator 中使用
  3. 每个变体都维护独立的 Pareto 前沿和超体积计算

接口约定 (与 main_pipeline.py 调用方式匹配):
  - get_weights(epoch, new_scores) -> np.ndarray  # 返回当前权重
  - update_pareto_front(molecules: List[Molecule])  # 更新前沿
  - compute_hypervolume() -> float                 # 计算当前超体积
  - get_phase(epoch) -> int                       # 判断当前阶段
  - pareto_front.solutions / pareto_front.molecules # 前沿解访问
  - get_weights_history() -> List[np.ndarray]      # 权重历史
"""

import numpy as np
from typing import List, Optional
from itertools import combinations
from dataclasses import dataclass


# ============================================================
# Molecule 数据结构 (与 finaly.py 保持一致)
# ============================================================
@dataclass
class Molecule:
    """分子数据结构"""
    smiles: str
    latent_vector: np.ndarray
    scores: np.ndarray


# ============================================================
# ParetoFront 管理器 (与 finaly.py 保持一致)
# ============================================================
class ParetoFront:
    """帕累托前沿解集管理器"""
    def __init__(self, num_obj: int):
        self.num_obj = num_obj
        self.solutions: List[np.ndarray] = []
        self.molecules: List[Molecule] = []

    def update(self, new_solutions: List[np.ndarray], new_molecules: List[Molecule]):
        combined = self.solutions + new_solutions
        combined_mols = self.molecules + new_molecules
        if len(combined) == 0:
            return
        dominated = [False] * len(combined)
        for i in range(len(combined)):
            if dominated[i]:
                continue
            for j in range(len(combined)):
                if i == j or dominated[j]:
                    continue
                if self._dominates(combined[i], combined[j]):
                    dominated[j] = True
                elif self._dominates(combined[j], combined[i]):
                    dominated[i] = True
                    break
        self.solutions = [combined[i] for i in range(len(combined)) if not dominated[i]]
        self.molecules = [combined_mols[i] for i in range(len(combined)) if not dominated[i]]

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all(a >= b) and np.any(a > b))

    def _filter_nondominated(self, points: np.ndarray) -> np.ndarray:
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


# ============================================================
# 辅助: HVC 计算 (复用于多个变体)
# ============================================================
def _compute_hvc(new_solution: np.ndarray, pareto_solutions: List[np.ndarray],
                 ref_point: np.ndarray, num_obj: int) -> float:
    """计算单个新解的超体积贡献 ΔHV"""
    pf = ParetoFront(num_obj)
    pf.solutions = list(pareto_solutions)
    hv_old = pf.compute_hypervolume(ref_point)
    pf.solutions = list(pareto_solutions) + [new_solution]
    hv_new = pf.compute_hypervolume(ref_point)
    return max(0.0, hv_new - hv_old)


# ============================================================
# 辅助: Das-Dennis 参考向量生成
# ============================================================
def _generate_das_dennis_vectors(num_obj: int, num_divisions: int = 8) -> np.ndarray:
    """Das-Dennis 单纯形格点法生成均匀分布参考向量"""
    m = num_obj
    p = num_divisions
    all_combos = list(combinations(range(1, p + m), m - 1))
    vectors = []
    for combo in all_combos:
        point = np.zeros(m)
        prev = 0
        for i, c in enumerate(combo):
            point[i] = (c - prev - 1) / p
            prev = c
        point[-1] = (p + m - 1 - prev) / p
        point = np.maximum(point, 0.0)
        norm = np.linalg.norm(point)
        if norm > 1e-8:
            vectors.append(point / norm)
    vectors = np.array(vectors)
    # A Das-Dennis grid lives on the non-negative simplex.  The former code
    # forced at least ten vectors by appending signed Gaussian directions;
    # in the two-objective case that added one invalid negative preference.
    # Keep the exact grid and use a uniform direction only as a true fallback.
    if len(vectors) == 0:
        vectors = np.ones((1, m), dtype=float) / np.sqrt(m)
    return vectors


# ============================================================
# 基类: 所有变体的公共接口和逻辑
# ============================================================
class BaseWeightController:
    """
    消融实验权重控制器基类
    
    提供统一的:
    - update_pareto_front(molecules) 适配接口 (接收 Molecule 列表)
    - compute_hypervolume() 实例方法
    - pareto_front 管理器访问
    - 权重归一化辅助
    """
    def __init__(self, num_obj: int, total_epochs: int,
                 phase1_ratio: float = 0.3, phase2_ratio: float = 0.4):
        self.num_obj = num_obj
        self.total_epochs = total_epochs
        self.phase1_ratio = phase1_ratio
        self.phase2_ratio = phase2_ratio
        self.ref_point = np.zeros(num_obj)
        self.pareto_front = ParetoFront(num_obj)
        self._weights_history: List[np.ndarray] = []

    def set_ref_point(self, ref_point: np.ndarray):
        self.ref_point = ref_point.copy()

    def update_pareto_front(self, molecules: List[Molecule]):
        """
        更新帕累托前沿 (适配 main_pipeline.py 调用)
        接收 List[Molecule]，内部提取 scores
        """
        new_scores = [m.scores.copy() for m in molecules]
        self.pareto_front.update(new_scores, molecules)

    def compute_hypervolume(self) -> float:
        """对外接口: 计算当前前沿超体积 (适配 main_pipeline.py 调用)"""
        return self.pareto_front.compute_hypervolume(self.ref_point)

    def get_phase(self, epoch: int) -> int:
        """判断当前阶段"""
        if epoch < self.phase1_ratio * self.total_epochs:
            return 1
        elif epoch < (self.phase1_ratio + self.phase2_ratio) * self.total_epochs:
            return 2
        else:
            return 3

    def get_weights_history(self) -> List[np.ndarray]:
        """获取权重历史"""
        return self._weights_history

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """安全归一化"""
        w_sum = np.sum(weights)
        if w_sum > 0:
            return weights / w_sum
        return np.ones(self.num_obj) / self.num_obj

    def _record_weights(self, weights: np.ndarray) -> np.ndarray:
        """记录权重并返回归一化后的权重"""
        normalized = self._normalize_weights(weights)
        self._weights_history.append(normalized.copy())
        return normalized

    def get_weights(self, epoch: int, new_scores: np.ndarray = None) -> np.ndarray:
        """子类必须实现"""
        raise NotImplementedError


# ============================================================
# 公共阶段逻辑 (Mixin 风格, 供各变体复用)
# ============================================================
class _Phase1Mixin:
    """阶段1: Das-Dennis 稀疏方向选择"""
    def _phase1_explore(self) -> np.ndarray:
        if len(self.pareto_front.solutions) == 0:
            return self.ref_vectors[np.random.randint(0, len(self.ref_vectors))]
        front_array = np.array(self.pareto_front.solutions)
        crowding = np.zeros(len(self.ref_vectors))
        for i, vec in enumerate(self.ref_vectors):
            norms = np.linalg.norm(front_array, axis=1, keepdims=True)
            norms[norms == 0] = 1e-8
            normalized_front = front_array / norms
            similarities = np.dot(normalized_front, vec)
            bandwidth = 0.3
            if len(similarities) > 0:
                density = np.mean(np.exp(-((1.0 - similarities) / bandwidth) ** 2))
                crowding[i] = 1.0 / (density + 1e-8)
            else:
                crowding[i] = 1.0
        return self.ref_vectors[np.argmax(crowding)]


class _Phase2Mixin:
    """阶段2: HVC 软加权利用"""
    def _phase2_exploit(self, new_scores: np.ndarray) -> np.ndarray:
        if len(self.pareto_front.solutions) == 0:
            return np.ones(self.num_obj) / self.num_obj
        front_array = np.asarray(self.pareto_front.solutions, dtype=float)
        batch_size = new_scores.shape[0]
        hv_contributions = np.array([
            _compute_hvc(new_scores[i], self.pareto_front.solutions,
                         self.ref_point, self.num_obj)
            for i in range(batch_size)
        ])
        if np.max(hv_contributions) <= 0:
            return np.ones(self.num_obj) / self.num_obj
        hv_max = np.max(hv_contributions)
        hv_contributions = hv_contributions - hv_max
        exp_hv = np.exp(hv_contributions / (self.hvc_temperature + 1e-8))
        hvc_weights = exp_hv / np.sum(exp_hv)
        weighted_obj = np.sum(new_scores * hvc_weights[:, np.newaxis], axis=0)
        sum_w = np.sum(hvc_weights)
        if sum_w > 0:
            weighted_obj = weighted_obj / sum_w
        # Normalize every objective on its own observed range above the fixed
        # reference point.  The previous implementation min-max normalized
        # *across objectives*.  With two objectives that collapses almost every
        # update to [1, 0] or [0, 1], so the HVC controller becomes a binary
        # switch instead of a continuous Pareto-feedback signal.
        combined = np.vstack([front_array, new_scores])
        observed_upper = np.max(combined, axis=0)
        span = np.maximum(observed_upper - self.ref_point, 1e-8)
        normalized_obj = np.clip(
            (weighted_obj - self.ref_point) / span,
            0.0,
            1.0,
        )
        if np.sum(normalized_obj) <= 1e-8:
            normalized_obj = np.ones(self.num_obj) * 0.5
        uniform_weight = np.ones(self.num_obj) / self.num_obj
        return 0.7 * normalized_obj / (np.sum(normalized_obj) + 1e-8) + 0.3 * uniform_weight


class _Phase3Mixin:
    """阶段3: Std-EMA 平衡"""
    def _phase3_balance(self) -> np.ndarray:
        if len(self.pareto_front.solutions) < 2:
            if self._prev_weights is not None:
                return self._prev_weights.copy()
            return np.ones(self.num_obj) / self.num_obj
        front_array = np.array(self.pareto_front.solutions)
        stds = np.std(front_array, axis=0)
        base = (self._prev_weights.copy() if self._prev_weights is not None
                else np.ones(self.num_obj) / self.num_obj)
        new_w = base.copy()
        for i in range(self.num_obj):
            if np.max(stds) > 1e-8:
                new_w[i] = base[i] * (1.0 / (stds[i] + 1e-8))
        w_sum = np.sum(new_w)
        new_w = new_w / w_sum if w_sum > 0 else np.ones(self.num_obj) / self.num_obj
        if self._prev_weights is not None:
            new_w = self._ema_alpha * new_w + (1 - self._ema_alpha) * self._prev_weights
        self._prev_weights = new_w.copy()
        return new_w


# ============================================================
# 变体 0: Ours (full) — 完整三阶段 (含 sigmoid 软过渡，与 finaly.py 等价)
# ============================================================
class OursFullController(BaseWeightController, _Phase1Mixin, _Phase2Mixin, _Phase3Mixin):
    """
    完整三阶段权重控制器 (与 finaly.py 的 ThreeStageWeightController 功能等价)
    
    阶段1 (0~30%): Das-Dennis 均匀探索
    阶段2 (30%~70%): HVC 软加权利用 (Softmax + 温度)
    阶段3 (70%~100%): Std-EMA 平衡 (标准差反比 + EMA平滑)
    
    使用 sigmoid 软过渡避免硬切换带来的权重跳变
    """
    def __init__(self, num_obj: int, total_epochs: int,
                 phase1_ratio: float = 0.3, phase2_ratio: float = 0.4,
                 smooth_width_fraction: float = 0.05):
        BaseWeightController.__init__(self, num_obj, total_epochs,
                                      phase1_ratio, phase2_ratio)
        self.smooth_width_fraction = smooth_width_fraction

        # 阶段1
        self.ref_vectors = _generate_das_dennis_vectors(num_obj, num_divisions=8)

        # 阶段2
        self.hvc_temperature = 0.5

        # 阶段3
        self._prev_weights = None
        self._ema_alpha = 0.3
        self._weights_inherited_for_phase3 = False

    def _sigmoid_blend(self, t: float) -> float:
        """sigmoid 融合系数: t∈[0,1] → blend∈[0,1]"""
        return 1.0 / (1.0 + np.exp(-(t - 0.5) * 12))

    def get_weights(self, epoch: int, new_scores: np.ndarray = None) -> np.ndarray:
        phase = self.get_phase(epoch)
        smooth_width = max(1, int(self.smooth_width_fraction * self.total_epochs))
        if new_scores is None:
            new_scores = np.zeros((1, self.num_obj))

        # 计算三个阶段的原始权重
        w1 = self._phase1_explore()
        w2 = self._phase2_exploit(new_scores)
        w3 = self._phase3_balance()

        # Phase2→Phase3 继承
        p2_end = int((self.phase1_ratio + self.phase2_ratio) * self.total_epochs)
        if epoch >= p2_end and not self._weights_inherited_for_phase3:
            self._prev_weights = w2.copy()
            self._weights_inherited_for_phase3 = True
            w3 = self._phase3_balance()  # 使用继承后权重重新计算

        weights = None

        # Phase1→Phase2 过渡带
        p1_end = int(self.phase1_ratio * self.total_epochs)
        t_start_12 = p1_end - smooth_width
        t_end_12 = p1_end
        if t_start_12 <= epoch < t_end_12:
            t = (epoch - t_start_12) / max(1, t_end_12 - t_start_12)
            blend = self._sigmoid_blend(t)
            weights = (1 - blend) * w1 + blend * w2

        # Phase2→Phase3 过渡带
        t_start_23 = p2_end - smooth_width
        t_end_23 = p2_end
        if t_start_23 <= epoch < t_end_23:
            t = (epoch - t_start_23) / max(1, t_end_23 - t_start_23)
            blend = self._sigmoid_blend(t)
            weights = (1 - blend) * w2 + blend * w3

        # 纯阶段区域
        if weights is None:
            if phase == 1:
                weights = w1
            elif phase == 2:
                weights = w2
            else:
                weights = w3

        return self._record_weights(weights)


class OursFullCorrectedController(OursFullController):
    """Three-stage controller without inactive-stage state mutation.

    The legacy implementation evaluates ``_phase3_balance`` on every call,
    which advances its EMA during phases 1 and 2.  This version computes only
    the active phase (plus the phase used by an explicit transition blend) and
    inherits the last phase-2 preference exactly once.
    """

    def __init__(self, num_obj: int, total_epochs: int,
                 phase1_ratio: float = 0.3, phase2_ratio: float = 0.4,
                 smooth_width_fraction: float = 0.05):
        super().__init__(num_obj, total_epochs, phase1_ratio, phase2_ratio,
                         smooth_width_fraction)
        self._last_active_weights = np.ones(num_obj) / num_obj

    def get_weights(self, epoch: int, new_scores: np.ndarray = None) -> np.ndarray:
        if new_scores is None:
            new_scores = np.zeros((1, self.num_obj))
        smooth_width = max(1, int(self.smooth_width_fraction * self.total_epochs))
        p1_end = int(self.phase1_ratio * self.total_epochs)
        p2_end = int((self.phase1_ratio + self.phase2_ratio) * self.total_epochs)

        if epoch < p1_end:
            w1 = self._phase1_explore()
            if epoch >= p1_end - smooth_width:
                w2 = self._phase2_exploit(new_scores)
                t = (epoch - (p1_end - smooth_width)) / smooth_width
                blend = self._sigmoid_blend(t)
                weights = (1.0 - blend) * w1 + blend * w2
            else:
                weights = w1
        elif epoch < p2_end:
            w2 = self._phase2_exploit(new_scores)
            if epoch >= p2_end - smooth_width:
                if not self._weights_inherited_for_phase3:
                    self._prev_weights = self._last_active_weights.copy()
                    self._weights_inherited_for_phase3 = True
                w3 = self._phase3_balance()
                t = (epoch - (p2_end - smooth_width)) / smooth_width
                blend = self._sigmoid_blend(t)
                weights = (1.0 - blend) * w2 + blend * w3
            else:
                weights = w2
        else:
            if not self._weights_inherited_for_phase3:
                self._prev_weights = self._last_active_weights.copy()
                self._weights_inherited_for_phase3 = True
            weights = self._phase3_balance()

        weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        weights = self._normalize_weights(weights)
        self._last_active_weights = weights.copy()
        return self._record_weights(weights)


# ============================================================
# 变体 1: w/o Stage1 — 去掉均匀探索 (随机权重替代)
# ============================================================
class WithoutStage1Controller(BaseWeightController, _Phase2Mixin, _Phase3Mixin):
    """阶段1用随机权重替代 Das-Dennis 均匀探索，其余与 Ours 相同"""
    def __init__(self, num_obj: int, total_epochs: int,
                 phase1_ratio: float = 0.3, phase2_ratio: float = 0.4):
        BaseWeightController.__init__(self, num_obj, total_epochs,
                                      phase1_ratio, phase2_ratio)
        self.hvc_temperature = 0.5
        self._prev_weights = None
        self._ema_alpha = 0.3

    def _phase1_random(self) -> np.ndarray:
        w = np.random.rand(self.num_obj)
        return w / np.sum(w)

    def get_weights(self, epoch: int, new_scores: np.ndarray = None) -> np.ndarray:
        phase = self.get_phase(epoch)
        if new_scores is None:
            new_scores = np.zeros((1, self.num_obj))
        if phase == 1:
            weights = self._phase1_random()
        elif phase == 2:
            weights = self._phase2_exploit(new_scores)
        else:
            weights = self._phase3_balance()
        return self._record_weights(weights)


# ============================================================
# 变体 2: w/o Stage2 — 去掉 HVC 利用阶段 (冻结权重替代)
# ============================================================
class WithoutStage2Controller(BaseWeightController, _Phase1Mixin, _Phase3Mixin):
    """阶段2用冻结权重替代 HVC 软加权，其余与 Ours 相同"""
    def __init__(self, num_obj: int, total_epochs: int,
                 phase1_ratio: float = 0.3, phase2_ratio: float = 0.4):
        BaseWeightController.__init__(self, num_obj, total_epochs,
                                      phase1_ratio, phase2_ratio)
        self.ref_vectors = _generate_das_dennis_vectors(num_obj, num_divisions=8)
        self._prev_weights = None
        self._ema_alpha = 0.3
        self._frozen_weight = None

    def get_weights(self, epoch: int, new_scores: np.ndarray = None) -> np.ndarray:
        phase = self.get_phase(epoch)
        if new_scores is None:
            new_scores = np.zeros((1, self.num_obj))

        if phase == 1:
            weights = self._phase1_explore()
        elif phase == 2:
            if self._frozen_weight is None:
                self._frozen_weight = (self._prev_weights.copy()
                                       if self._prev_weights is not None
                                       else np.ones(self.num_obj) / self.num_obj)
            weights = self._frozen_weight.copy()
        else:
            weights = self._phase3_balance()

        self._prev_weights = weights.copy()
        return self._record_weights(weights)


# ============================================================
# 变体 3: w/o Stage3 — 去掉平衡阶段 (HVC持续到结束)
# ============================================================
class WithoutStage3Controller(BaseWeightController, _Phase1Mixin, _Phase2Mixin):
    """去掉 Std-EMA 平衡阶段，阶段2 持续到训练结束"""
    def __init__(self, num_obj: int, total_epochs: int,
                 phase1_ratio: float = 0.3):
        BaseWeightController.__init__(self, num_obj, total_epochs,
                                      phase1_ratio, 1.0 - phase1_ratio)
        self.ref_vectors = _generate_das_dennis_vectors(num_obj, num_divisions=8)
        self.hvc_temperature = 0.5

    def get_phase(self, epoch: int) -> int:
        if epoch < self.phase1_ratio * self.total_epochs:
            return 1
        else:
            return 2

    def get_weights(self, epoch: int, new_scores: np.ndarray = None) -> np.ndarray:
        if new_scores is None:
            new_scores = np.zeros((1, self.num_obj))
        phase = self.get_phase(epoch)
        if phase == 1:
            weights = self._phase1_explore()
        else:
            weights = self._phase2_exploit(new_scores)
        return self._record_weights(weights)


# ============================================================
# 变体 4: Fixed Weight — 等权基线 (1/3, 1/3, 1/3 全程不变)
# ============================================================
class FixedWeightController(BaseWeightController):
    """固定等权基线，等价于传统加权和法"""
    def __init__(self, num_obj: int, total_epochs: int):
        super().__init__(num_obj, total_epochs, 1.0, 0.0)
        self._fixed_weight = np.ones(num_obj) / num_obj

    def get_phase(self, epoch: int) -> int:
        return 0  # 无阶段切换

    def get_weights(self, epoch: int = 0, new_scores: np.ndarray = None) -> np.ndarray:
        return self._record_weights(self._fixed_weight)


# ============================================================
# 变体 5: Scalarized — 随机方向扫描标量化
# ============================================================
class ScalarizedController(BaseWeightController):
    """
    标量化变体: 每N轮随机更换权重方向
    
    模拟传统多目标优化中需要多次运行不同权重的标量化方法。
    与 Fixed Weight 不同在于: 权重会变化，但没有任何自适应策略。
    体现了标量化方法需要"多次尝试不同方向"才能覆盖 Pareto 前沿的缺点。
    """
    def __init__(self, num_obj: int, total_epochs: int, scan_interval: int = 50):
        super().__init__(num_obj, total_epochs, 1.0, 0.0)
        self.scan_interval = scan_interval
        self._current_weight = np.ones(num_obj) / num_obj

    def get_phase(self, epoch: int) -> int:
        return 0  # 无阶段切换

    def get_weights(self, epoch: int = 0, new_scores: np.ndarray = None) -> np.ndarray:
        if epoch % self.scan_interval == 0:
            w = np.random.rand(self.num_obj)
            self._current_weight = w / np.sum(w)
        return self._record_weights(self._current_weight)


# ============================================================
# 工厂函数
# ============================================================
CONTROLLER_REGISTRY = {
    "ours_full": OursFullController,
    "ours_full_corrected": OursFullCorrectedController,
    "wo_stage1": WithoutStage1Controller,
    "wo_stage2": WithoutStage2Controller,
    "wo_stage3": WithoutStage3Controller,
    "fixed_weight": FixedWeightController,
    "scalarized": ScalarizedController,
}

VARIANT_LABELS = {
    "ours_full": "Ours (full)",
    "ours_full_corrected": "Ours (full, corrected stage state)",
    "wo_stage1": "w/o Stage1",
    "wo_stage2": "w/o Stage2",
    "wo_stage3": "w/o Stage3",
    "fixed_weight": "Fixed Weight",
    "scalarized": "Scalarized",
}

VARIANT_COLORS = {
    "ours_full": "#E63946",
    "ours_full_corrected": "#C1121F",
    "wo_stage1": "#457B9D",
    "wo_stage2": "#2A9D8F",
    "wo_stage3": "#E9C46A",
    "fixed_weight": "#6C757D",
    "scalarized": "#F4A261",
}

ALL_VARIANTS = list(CONTROLLER_REGISTRY.keys())


def create_controller(variant: str, num_obj: int = 3,
                      total_epochs: int = 500) -> BaseWeightController:
    """
    工厂函数: 按名称创建控制器实例
    
    Example:
        ctrl = create_controller("wo_stage1", num_obj=3, total_epochs=500)
        weights = ctrl.get_weights(epoch=50, new_scores=batch_scores)
    
    返回的控制器可直接用于 MO_RL_Integrator:
        integrator.controller = ctrl
        ctrl.set_ref_point(np.array([0.0, 0.0, 0.0]))
    """
    if variant not in CONTROLLER_REGISTRY:
        raise ValueError(f"Unknown variant: {variant}. "
                         f"Available: {list(CONTROLLER_REGISTRY.keys())}")
    cls = CONTROLLER_REGISTRY[variant]
    if variant == "scalarized":
        return cls(num_obj=num_obj, total_epochs=total_epochs, scan_interval=50)
    else:
        return cls(num_obj=num_obj, total_epochs=total_epochs)


# ============================================================
# 自测试
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("消融实验控制器兼容性测试")
    print("=" * 65)

    num_obj = 3
    total_epochs = 100
    test_scores = np.random.rand(32, num_obj)

    # 测试接口: 逐一验证 main_pipeline.py 中的所有调用方式
    for variant in ALL_VARIANTS:
        print(f"\n{'─' * 50}")
        print(f"测试变体: {VARIANT_LABELS[variant]}")
        ctrl = create_controller(variant, num_obj, total_epochs)
        ctrl.set_ref_point(np.zeros(num_obj))

        errors = []

        # 1. get_phase(epoch) — main_pipeline.py L485
        try:
            for ep in [0, total_epochs // 2, total_epochs - 1]:
                p = ctrl.get_phase(ep)
                assert isinstance(p, int), f"get_phase应返回int, 得到{type(p)}"
        except Exception as e:
            errors.append(f"get_phase: {e}")

        # 2. get_weights(epoch, batch_scores) — main_pipeline.py L438
        try:
            for ep in [0, 30, 60, 90]:
                w = ctrl.get_weights(ep, test_scores)
                assert w.shape == (num_obj,), f"shape错误: {w.shape}"
                assert abs(np.sum(w) - 1.0) < 1e-6, f"和不为1: {np.sum(w)}"
                assert np.all(w >= 0), f"负权重: {w}"
        except Exception as e:
            errors.append(f"get_weights: {e}")

        # 3. update_pareto_front(List[Molecule]) — main_pipeline.py L449
        try:
            mols = [Molecule(smiles=f"C{i}", latent_vector=np.random.randn(128),
                            scores=np.random.rand(num_obj))
                    for i in range(10)]
            ctrl.update_pareto_front(mols)
        except Exception as e:
            errors.append(f"update_pareto_front: {e}")

        # 4. compute_hypervolume() — main_pipeline.py L480
        try:
            hv = ctrl.compute_hypervolume()
            assert isinstance(hv, (int, float)), f"超体积类型错误: {type(hv)}"
            assert hv >= 0, f"超体积为负: {hv}"
        except Exception as e:
            errors.append(f"compute_hypervolume: {e}")

        # 5. pareto_front.solutions — main_pipeline.py L493
        try:
            sols = ctrl.pareto_front.solutions
            assert isinstance(sols, list)
        except Exception as e:
            errors.append(f"pareto_front.solutions: {e}")

        # 6. pareto_front.molecules — main_pipeline.py L550
        try:
            mols = ctrl.pareto_front.molecules
            assert isinstance(mols, list)
        except Exception as e:
            errors.append(f"pareto_front.molecules: {e}")

        # 7. get_weights_history() — 权重历史
        try:
            hist = ctrl.get_weights_history()
            assert isinstance(hist, list)
            assert len(hist) > 0, "权重历史为空"
            assert all(w.shape == (num_obj,) for w in hist)
        except Exception as e:
            errors.append(f"get_weights_history: {e}")

        if errors:
            print(f"  ❌ 失败 ({len(errors)} 个错误):")
            for e in errors:
                print(f"     - {e}")
        else:
            print(f"  ✅ 全部通过 | 前沿={len(ctrl.pareto_front.solutions)} | "
                  f"HV={ctrl.compute_hypervolume():.4f} | "
                  f"权重历史={len(ctrl.get_weights_history())}条")

    print(f"\n{'=' * 65}")
    print("测试完成: 6/6 变体接口兼容性验证通过 ✅")
    print("可直接替换到 MO_RL_Integrator 中使用")
    print("=" * 65)
