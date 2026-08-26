"""
main_integrator.py - 三阶段 MO-RL 核心集成程序

功能：整合 PPO 代理、动态权重控制器和 VAE 解码器。
流程：隐空间采样 -> PPO 优化 -> 动态权重调整 -> 分子生成与评估。
"""
import os
import subprocess
import sys

# ==================== Conda 环境自动激活 ====================
def activate_conda_env(env_name: str = "myenv"):
    """自动激活conda虚拟环境"""
    # 检查当前是否已在目标环境中
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env == env_name:
        return True  # 已在目标环境
    
    # 尝试重新运行脚本
    try:
        # 使用conda run启动当前脚本
        result = subprocess.run(
            ["conda", "run", "-n", env_name, sys.executable] + sys.argv,
            capture_output=False
        )
        sys.exit(result.returncode)
    except Exception:
        pass
    
    # 检查是否安装了conda
    try:
        subprocess.run(["conda", "--version"], capture_output=True, check=True)
        print(f"⚠️ 请手动激活环境: conda activate {env_name}")
    except Exception:
        print(f"⚠️ 未检测到Conda，请安装Miniconda或设置正确的Python环境")
    
    return False

# 尝试激活conda环境 (可选 - 注释掉以禁用自动激活)
# if not activate_conda_env("myenv"):
#     pass  # 脚本将在激活的环境中被重新调用

# 继续正常导入
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import csv
import pandas as pd
import sys
import importlib.util
from ppo_agent import PPOAgent
from finaly import ThreeStageWeightController, Molecule
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED
import random
from typing import List, Tuple

# ==================== Polygon VAE 动态加载 ====================
# 直接导入 vae_model 模块，绕过 __init__.py 中的 joblib 等可选依赖
def _get_polygon_vae_class():
    """返回 Polygon VAE 类，通过直接加载 vae_model.py 绕过可选依赖"""
    try:
        from polygon.vae.vae_model import VAE
        return VAE
    except Exception:
        pass

    polygon_vae_path = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "polygon-main"
        / "polygon"
        / "vae"
        / "vae_model.py"
    )
    if not polygon_vae_path.exists():
        raise FileNotFoundError(f"Polygon VAE 模块未找到: {polygon_vae_path}")
    
    spec = importlib.util.spec_from_file_location(
        "polygon_vae_model", str(polygon_vae_path)
    )
    vm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vm)
    return vm.VAE

def _infer_polygon_config(state_dict: dict) -> dict:
    """从 Polygon VAE 的 state_dict 推断模型配置 (参考 check_single.py)"""
    config = {}
    config['q_bidir'] = any('_reverse' in k for k in state_dict.keys())
    w_hh = state_dict['encoder_rnn.weight_hh_l0']
    config['q_d_h'] = w_hh.size(1)
    layer_indices = []
    for k in state_dict.keys():
        if k.startswith('decoder_rnn.weight_ih_l'):
            layer_indices.append(int(k.split('_l')[-1]))
    config['d_n_layers'] = max(layer_indices) + 1 if layer_indices else 3
    w_hh_dec = state_dict['decoder_rnn.weight_hh_l0']
    config['d_d_h'] = w_hh_dec.size(1)
    config['d_z'] = state_dict['decoder_lat.weight'].size(1)
    config['q_cell'] = 'gru'
    config['d_cell'] = 'gru'
    config['q_n_layers'] = 1
    config['q_dropout'] = 0.5
    config['d_dropout'] = 0.2
    config['freeze_embeddings'] = False
    return config


def _is_polygon_model(state_dict: dict) -> bool:
    """检测 state_dict 是否来自 Polygon VAE（有内置词汇表）
    
    Polygon VAE 的特征: state_dict 中包含 'x_emb.weight'（Embedding 权重）
    并且不包含旧版模型的固定参数名（如特定 Bos/ EOS 约定）。
    更可靠的判断: 检查是否有 'encoder'/'decoder' 的 ModuleList 结构，
    或者直接检查 vocab 相关标志。
    这里根据 Polygon VAE 的命名约定: 没有 external vocab 依赖
    最简单的区分: Polygon 模型键名不含外部vocab参数，Legacy 模型通过
    state_dict 键特征来判断。
    """
    # Polygon VAE 的关键特征: vae.encoder, vae.decoder (ModuleList)
    # 且 decoder_rnn 的输入维度 = d_emb + d_z (而非固定 emb_dim=51)
    if 'vae.encoder' in state_dict:
        return True
    # 如果 state_dict 是扁平键名格式 (加载后没有vae.前缀)
    # 通过 decoder_rnn.weight_ih_l0 的输入维度判断
    if 'decoder_rnn.weight_ih_l0' in state_dict:
        w_ih = state_dict['decoder_rnn.weight_ih_l0']
        # Polygon VAE: input_size = d_emb + d_z (通常是 51+128=179)
        # Legacy VAEModel: input_size = emb_dim + latent_dim (51+128=179)
        # 两者结构一致，但 Polygon 没有模型封装在 dict 中
        # 进一步检查是否有 'x_emb.weight' 和 encoder
        return True  # 默认按 Polygon 处理 (新版优先)
    return False


# ==================== 旧版 VAE 解码器 (Legacy) ====================
class VAEModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 51, enc_hidden: int = 256,
                 latent_dim: int = 128, dec_hidden: int = 512, dec_layers: int = 3):
        super(VAEModel, self).__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.dec_hidden = dec_hidden
        self.dec_layers = dec_layers

        self.x_emb = nn.Embedding(vocab_size, emb_dim)
        self.encoder_rnn = nn.GRU(input_size=emb_dim, hidden_size=enc_hidden,
                                  batch_first=True)
        self.q_mu = nn.Linear(enc_hidden, latent_dim)
        self.q_logvar = nn.Linear(enc_hidden, latent_dim)

        self.decoder_lat = nn.Linear(latent_dim, dec_hidden)
        self.decoder_rnn = nn.GRU(input_size=emb_dim + latent_dim,
                                  hidden_size=dec_hidden,
                                  num_layers=dec_layers,
                                  batch_first=True)
        self.decoder_fc = nn.Linear(dec_hidden, vocab_size)

    def decode_latent(self, z: torch.Tensor, max_length: int = 120,
                      bos_id: int = 48, eos_id: int = 49, greedy: bool = False,
                      temperature: float = 0.7) -> torch.Tensor:
        batch_size = z.shape[0]
        hidden = self.decoder_lat(z).unsqueeze(0).repeat(self.dec_layers, 1, 1)
        next_token = torch.full((batch_size, 1), bos_id, dtype=torch.long,
                                device=z.device)
        generated = []

        for _ in range(max_length):
            emb = self.x_emb(next_token)
            latent_expanded = z.unsqueeze(1).expand(-1, 1, -1)
            rnn_input = torch.cat([emb, latent_expanded], dim=-1)
            out, hidden = self.decoder_rnn(rnn_input, hidden)
            logits = self.decoder_fc(out.squeeze(1))
            
            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            
            generated.append(next_token)
            if torch.all(next_token == eos_id):
                break

        if len(generated) == 0:
            return torch.full((batch_size, 1), eos_id, dtype=torch.long,
                               device=z.device)
        return torch.cat(generated, dim=1)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("VAEModel only supports decode_latent for now.")


# ==================== VAE 解码器接口 (双模式: Polygon + Legacy) ====================
class VAE_Decoder:
    """
    VAE 解码器接口，支持两种模型:
    - Polygon VAE (内置词汇表，model.sample() 直接返回 SMILES)
    - Legacy VAEModel (外部词汇表，decode_latent -> token IDs -> ids2string)
    
    自动检测模型类型，无需手动指定。
    
    优化特性:
    - 批量解码支持
    - SMILES验证缓存
    """
    def __init__(self, latent_dim: int = 128,
                 model_path: str = None,
                 vocab_path: str = None,
                 device: torch.device = None,
                 enable_cache: bool = True):
        self.latent_dim = latent_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 标记模型类型: 'polygon' 或 'legacy'
        self._model_type = None
        
        # SMILES验证缓存
        self._smiles_cache = {} if enable_cache else None
        self._cache_hits = 0
        
        if model_path is not None:
            self.load_model(model_path, vocab_path)
        else:
            raise ValueError("model_path is required for VAE_Decoder")

    def _load_polygon_model(self, model_path: str):
        """加载 Polygon VAE 模型（内置词汇表，无需外部vocab）"""
        device = self.device
        state_dict = torch.load(model_path, map_location=device)
        config = _infer_polygon_config(state_dict)
        print(f"  [Polygon VAE] 推断配置: bidir={config['q_bidir']}, "
              f"q_d_h={config['q_d_h']}, d_d_h={config['d_d_h']}, "
              f"d_z={config['d_z']}, d_n_layers={config['d_n_layers']}")
        
        VAE = _get_polygon_vae_class()
        self.model = VAE(**config)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()
        self._model_type = 'polygon'
        # Polygon VAE 内置词汇表，无需外部 vocab 对象
        self.vocab = None
        # 更新 latent_dim 为实际值
        self.latent_dim = config['d_z']
        print(f"  [Polygon VAE] 模型加载成功 (latent_dim={self.latent_dim})")

    def _load_legacy_model(self, model_path: str, vocab_path: str = None):
        """加载旧版 VAEModel（需要外部词汇表）"""
        device = self.device
        # 加载词汇表
        if vocab_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            vocab_path = base_dir / "vocab" / "twoG_smiles.pkl"
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        
        model_vocab_size = len(self.vocab) - 1 if len(self.vocab) > 0 else len(self.vocab)
        self.model = VAEModel(vocab_size=model_vocab_size, latent_dim=self.latent_dim).to(device)
        self.model.eval()
        
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and not isinstance(state, torch.nn.Module):
            filtered = {k: v for k, v in state.items() if k in self.model.state_dict()}
            self.model.load_state_dict(filtered, strict=False)
        elif isinstance(state, torch.nn.Module):
            self.model = state.to(device)
            self.model.eval()
        else:
            raise ValueError(f"Unsupported model file format: {type(state)}")
        
        self._model_type = 'legacy'
        print(f"  [Legacy VAE] 模型加载成功 (vocab_size={model_vocab_size})")

    def load_model(self, model_path: str, vocab_path: str = None):
        """自动检测并加载模型 (Polygon 或 Legacy)"""
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 判断模型类型
        # Polygon 模型的 state_dict 顶层有 'x_emb.weight' 和 encoder/decoder ModuleList 键
        # Legacy 模型加载后可能是多层嵌套的 dict
        # 最简单的方法: 尝试按 Polygon 方式加载，失败则回退到 Legacy
        try:
            self._load_polygon_model(model_path)
            print(f"  ✓ 检测到 Polygon VAE 模型")
        except Exception as e:
            print(f"  [Polygon 加载失败: {e}]，尝试 Legacy 模式...")
            self._load_legacy_model(model_path, vocab_path)
            print(f"  ✓ 使用 Legacy VAEModel")

    def _decode_polygon(self, z: torch.Tensor, greedy: bool = False,
                        temperature: float = 0.7) -> list:
        """Polygon VAE 解码: 调用 model.sample() 直接返回 SMILES"""
        n_batch = z.shape[0]
        temp = temperature if not greedy else 0.2  # greedy 时降低温度
        smiles_list = self.model.sample(
            n_batch=n_batch, z=z, max_len=120, temp=temp, multinomial=(not greedy)
        )
        return smiles_list

    def _decode_legacy(self, z: torch.Tensor, greedy: bool = False,
                       temperature: float = 0.7) -> list:
        """Legacy VAEModel 解码: decode_latent -> token IDs -> SMILES"""
        with torch.no_grad():
            token_ids = self.model.decode_latent(
                z, bos_id=self.vocab.bos, eos_id=self.vocab.eos, greedy=greedy,
                temperature=temperature
            )
            token_ids = token_ids.cpu().numpy().tolist()
        
        result = []
        filter_ids = {self.vocab.bos, self.vocab.eos, self.vocab.pad}
        for seq in token_ids:
            filtered = [int(i) for i in seq if int(i) not in filter_ids]
            result.append(self.vocab.ids2string(filtered))
        return result

    def decode(self, z: np.ndarray, greedy: bool = False, temperature: float = 0.7) -> str:
        """解码单个隐向量 (向后兼容)"""
        results = self.decode_batch(z.reshape(1, -1), greedy=greedy, temperature=temperature)
        return results[0] if results else ""

    def decode_batch(self, z: np.ndarray, greedy: bool = False,
                     temperature: float = 0.7) -> list:
        """批量解码多个隐向量
        
        Args:
            z: 隐向量数组 (batch_size, latent_dim)
            greedy: 是否使用greedy解码
            temperature: 采样温度 (默认0.7, 越高多样性越大)
        Returns:
            list: SMILES字符串列表
        """
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z.astype(np.float32)).to(self.device)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        
        if self._model_type == 'polygon':
            return self._decode_polygon(z, greedy=greedy, temperature=temperature)
        else:
            return self._decode_legacy(z, greedy=greedy, temperature=temperature)

    def _check_smiles_validity(self, smiles: str) -> bool:
        """快速SMILES有效性检查 (带缓存)"""
        if self._smiles_cache is None:
            return self._validate_smiles(smiles)
        
        if smiles in self._smiles_cache:
            self._cache_hits += 1
            return self._smiles_cache[smiles]
        
        is_valid = self._validate_smiles(smiles)
        if len(self._smiles_cache) < 10000:
            self._smiles_cache[smiles] = is_valid
        return is_valid
    
    def _validate_smiles(self, smiles: str) -> bool:
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False
    
    def get_cache_stats(self) -> dict:
        if self._smiles_cache is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "size": len(self._smiles_cache),
            "hits": self._cache_hits,
            "hit_rate": self._cache_hits / max(1, self._cache_hits + len(self._smiles_cache))
        }

    def sample_latent_space(self, num_samples: int) -> np.ndarray:
        return np.random.randn(num_samples, self.latent_dim)

# ==================== 目标函数计算器 ====================
class ObjectiveCalculator:
    """
    目标函数计算器
    功能：计算分子的各项属性得分（如 QED, EGFR, VEGFR2）。
    """
    def __init__(self,
                 egfr_model_path: str = None,
                 vegfr2_model_path: str = None,
                 fingerprint_size: int = 2048,
                 fingerprint_radius: int = 2):
        self.egfr_model = self._load_model(egfr_model_path)
        self.vegfr2_model = self._load_model(vegfr2_model_path)
        self.fingerprint_size = fingerprint_size
        self.fingerprint_radius = fingerprint_radius

    def _load_model(self, model_path: str):
        if model_path is None:
            return None
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def _smiles_to_fingerprint(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.fingerprint_size, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.fingerprint_radius,
            nBits=self.fingerprint_size,
            useChirality=True
        )
        arr = np.zeros((1, self.fingerprint_size), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr[0])
        return arr[0].astype(np.float32)

    def _safe_qed(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        try:
            return float(QED.qed(mol))
        except Exception:
            return 0.0


    def _apply_quality_penalties(self, smiles: str, egfr: float, vegfr2: float, qed: float):
        """应用分子质量惩罚机制，抑制低质量分子"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0, 0.0, 0.0
        
        penalty_factor = 1.0
        
        # 惩罚1: SMILES长度惩罚 (超过80字符开始惩罚)
        smiles_len = len(smiles)
        if smiles_len > 80:
            length_penalty = max(0.3, 1.0 - (smiles_len - 80) * 0.0175)
            penalty_factor *= length_penalty
        
        # 惩罚2: 检测过度重复的子结构
        max_repeat = 1
        current_repeat = 1
        for i in range(1, len(smiles)):
            if smiles[i] == smiles[i-1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1
        
        if max_repeat > 10:
            repeat_penalty = max(0.2, 1.0 - (max_repeat - 10) * 0.08)
            penalty_factor *= repeat_penalty
        
        return egfr * penalty_factor, vegfr2 * penalty_factor, qed * penalty_factor
    def calculate_scores(self, smiles: str) -> np.ndarray:
        """
        计算分子的三目标得分：QED、EGFR、VEGFR2。
        """
        qed_score = self._safe_qed(smiles)
        fingerprint = self._smiles_to_fingerprint(smiles)
        egfr_score = 0.0
        vegfr2_score = 0.0

        if self.egfr_model is not None:
            try:
                egfr_score = float(self.egfr_model.predict(fingerprint.reshape(1, -1))[0])
            except Exception:
                egfr_score = 0.0

        if self.vegfr2_model is not None:
            try:
                vegfr2_score = float(self.vegfr2_model.predict(fingerprint.reshape(1, -1))[0])
            except Exception:
                vegfr2_score = 0.0

        # 应用质量惩罚机制
        egfr_score, vegfr2_score, qed_score = self._apply_quality_penalties(
            smiles, egfr_score, vegfr2_score, qed_score
        )

        # 修复: 返回顺序必须与CSV列名[egfr, vegfr2, qed]一致
        return np.array([egfr_score, vegfr2_score, qed_score], dtype=np.float32)

# ==================== 核心训练循环 ====================
class MO_RL_Integrator:
    """多目标强化学习核心集成器 (优化版本)
    
    优化特性:
    - 参数化配置 (magic numbers可调整)
    - 内存管理 (GPU缓存清理)
    - 批量解码支持
    """
    # 默认参数 (可配置)
    DEFAULT_CONFIG = {
        "step_size": 0.12,           # 隐向量调整步长
        "z_clip_range": (-2.0, 2.0), # 隐向量裁剪范围
        "min_smiles_len": 5,         # 最小SMILES长度
        "max_smiles_len": 80,         # 最大SMILES长度
        "ppo_update_min_samples": 4,    # PPO更新的最小样本数
        "reward_clip_range": (-1.0, 1.0), # 奖励裁剪范围
        "use_greedy_decode": False,   # 使用greedy解码
        "temperature": 0.7,           # 采样温度 (默认0.7, 越高多样性越大)
        "checkpoint_freq": 10,        # checkpoint保存频率
        "dynamic_ref_point": False,    # 是否使用动态参考点
        "ref_point_margin": 0.05,     # 动态参考点边距
        # === Temperature Curriculum ===
        "use_temperature_curriculum": False,  # 是否启用温度退火
        "temperature_start": 1.5,      # 初始采样温度 (高→探索)
        "temperature_end": 0.3,        # 最终采样温度 (低→利用)
        "temperature_schedule": "cosine",  # 退火方式: cosine / linear / exponential
        # === Step Size Curriculum ===
        "use_step_size_curriculum": False,  # 是否启用步长退火
        "step_size_start": 0.25,       # 初始步长 (大→大范围探索)
        "step_size_end": 0.05,         # 最终步长 (小→精细调整)
        "step_size_schedule": "cosine",    # 退火方式
        # === Cosine Annealing LR (用于 PPO) ===
        "use_cosine_lr": False,        # 是否使用余弦退火学习率
        "lr_max": 3e-4,                # 最大学习率
        "lr_min": 1e-5,                # 最小学习率
        "lr_t_max": 1000,              # 余弦周期 (通常=total_epochs)
        # === Adapt PPO interface ===
        "use_entropy_bonus": False,    # 是否启用 Entropy Bonus
        "entropy_coef": 0.01,          # Entropy 系数
        "use_minibatch_updates": False, # 是否使用 Mini-batch 更新
        "minibatch_size": 16,          # Mini-batch 大小
        "ppo_update_epochs": 4,        # PPO 更新轮数
    }
    
    def __init__(self, 
                 latent_dim: int = 128, 
                 num_obj: int = 3, 
                 total_epochs: int = 1000,
                 batch_size: int = 64,
                 vae_model_path: str = None,
                 vocab_path: str = None,
                 egfr_model_path: str = None,
                 vegfr2_model_path: str = None,
                 config: dict = None):
        
        self.latent_dim = latent_dim
        self.num_obj = num_obj
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        
        # 合并配置
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        # 初始化组件
        self.vae = VAE_Decoder(latent_dim=latent_dim,
                               model_path=vae_model_path,
                               vocab_path=vocab_path)
        # VAE 加载后自动适配实际隐空间维度（兼容 128/512/自定义维度）
        actual_latent_dim = self.vae.latent_dim
        self.latent_dim = actual_latent_dim
        print(f"  ✓ MO-RL 隐空间维度适配: PPO网络输入={actual_latent_dim}")
        self.objective_calculator = ObjectiveCalculator(
            egfr_model_path=egfr_model_path,
            vegfr2_model_path=vegfr2_model_path
        )
        
        # 初始化 PPO 代理（使用 VAE 实际隐空间维度）
        self.agent = PPOAgent(state_dim=actual_latent_dim, action_dim=actual_latent_dim)
        
        # 初始化三阶段控制器
        self.controller = ThreeStageWeightController(num_obj, total_epochs)
        
        # 参考点设置
        self.ref_point = np.array([0.0, 0.0, 0.0])
        self.controller.set_ref_point(self.ref_point)
        
        # 训练统计
        self._total_valid_molecules = 0
        self._total_cache_hits = 0
        
        # 权重历史记录
        self._weight_history: List[Tuple[int, np.ndarray, float]] = []

    def _get_curriculum_value(self, epoch: int, total: int, start: float,
                               end: float, schedule: str = "cosine") -> float:
        """通用退火调度器"""
        if total <= 1:
            return end
        t = np.clip(epoch / (total - 1), 0.0, 1.0)
        if schedule == "cosine":
            return end + (start - end) * 0.5 * (1 + np.cos(np.pi * t))
        elif schedule == "linear":
            return start + (end - start) * t
        elif schedule == "exponential":
            ratio = end / start if start != 0 else 0.1
            return start * (ratio ** t)
        return end

    def _update_dynamic_ref_point(self) -> None:
        """基于当前 Pareto 前沿最差值动态更新参考点"""
        if not self.config["dynamic_ref_point"]:
            return
        solutions = self.controller.pareto_front.solutions
        if len(solutions) < 3:
            return
        arr = np.array(solutions)
        mins = arr.min(axis=0)
        margin = self.config["ref_point_margin"]
        ranges = arr.max(axis=0) - mins + 1e-8
        self.ref_point = mins - margin * ranges
        self.controller.set_ref_point(self.ref_point)

    def _get_cosine_lr(self, epoch: int) -> float:
        """余弦退火学习率"""
        cfg = self.config
        t = np.clip(epoch / max(1, cfg["lr_t_max"] - 1), 0.0, 1.0)
        return cfg["lr_min"] + (cfg["lr_max"] - cfg["lr_min"]) * 0.5 * (1 + np.cos(np.pi * t))

    def run_episode(self, epoch: int) -> Tuple[List[Molecule], float]:
        """
        运行一个完整的训练周期（Episode）
        
        使用config参数化配置
        """
        batch_molecules = []
        batch_experiences = []
        cfg = self.config
        
        # 调试统计
        debug_counts = {"total": 0, "len_fail": 0, "valid_fail": 0, "nan_fail": 0}
        
        # 1. 从隐空间采样初始状态
        z_states = self.vae.sample_latent_space(self.batch_size)
        
        # 2. 生成候选分子并收集经验
        
        # ---- Curriculum: 退火步长和温度 ----
        if cfg["use_temperature_curriculum"]:
            current_temperature = self._get_curriculum_value(
                epoch, self.total_epochs, cfg["temperature_start"],
                cfg["temperature_end"], cfg["temperature_schedule"])
        else:
            current_temperature = cfg["temperature"]
        
        if cfg["use_step_size_curriculum"]:
            current_step_size = self._get_curriculum_value(
                epoch, self.total_epochs, cfg["step_size_start"],
                cfg["step_size_end"], cfg["step_size_schedule"])
        else:
            current_step_size = cfg["step_size"]
        
        # ---- 动态参考点 ----
        self._update_dynamic_ref_point()
        
        for i in range(self.batch_size):
            debug_counts["total"] += 1
            state = z_states[i]
            action, log_prob, value, entropy_val = self.agent.select_action(state)
            
            # 使用退火后的步长
            new_z = np.clip(state + action * current_step_size, cfg["z_clip_range"][0], cfg["z_clip_range"][1])
            
            # 使用退火后的温度
            smiles = self.vae.decode(new_z, greedy=cfg["use_greedy_decode"],
                                     temperature=current_temperature)
            
            # 配置的长度验证
            if len(smiles) < cfg["min_smiles_len"] or len(smiles) > cfg["max_smiles_len"]:
                debug_counts["len_fail"] += 1
                continue
            
            # 使用验证缓存
            if not self.vae._check_smiles_validity(smiles):
                debug_counts["valid_fail"] += 1
                continue
                
            scores = self.objective_calculator.calculate_scores(smiles)
            
            # 检查分数有效性
            if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
                debug_counts["nan_fail"] += 1
                continue

            mol_obj = Molecule(smiles, new_z, scores)
            batch_molecules.append(mol_obj)
            batch_experiences.append((state, action, log_prob, value, scores, entropy_val))
        
        # 无有效分子则返回
        if len(batch_molecules) == 0:
            # 打印调试信息
            if epoch % 10 == 0:
                print(f"  ⚠️ Epoch {epoch}: 生成分子失败统计 - {debug_counts}")
            return [], 0.0
        
        # 更新统计
        self._total_valid_molecules += len(batch_molecules)
        
        # 3. 计算动态权重并存储加权奖励
        batch_scores = np.array([m.scores for m in batch_molecules])
        weights = self.controller.get_weights(epoch, batch_scores)
        
        # 使用配置的奖励裁剪范围
        for state, action, log_prob, value, scores, entropy_val in batch_experiences:
            reward = float(np.dot(scores, weights))
            reward = np.clip(reward, cfg["reward_clip_range"][0], cfg["reward_clip_range"][1])
            
            self.agent.store_transition(
                state=state,
                action=action,
                reward=reward,
                log_prob=float(log_prob),
                value=float(value),
                done=False,
                entropy=float(entropy_val)
            )

        # 4. PPO 代理更新 (使用配置的最小样本数)
        loss = 0.0
        if len(batch_molecules) >= cfg["ppo_update_min_samples"]:
            try:
                loss = self.agent.update()
            except Exception as e:
                print(f"  ⚠️ PPO更新失败: {e}")

        # 5. 更新帕累托前沿
        self.controller.update_pareto_front(batch_molecules)
        
        # 定期清理GPU缓存
        if epoch % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return batch_molecules, loss

    def train(self):
        """
        主训练循环（带实时进度条）
        """
        import time
        print("🚀 开始三阶段多目标强化学习训练...")
        
        # 创建日志目录
        log_dir = Path(__file__).resolve().parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        t_start = time.time()
        bar_width = 30  # 进度条宽度
        
        for epoch in range(self.total_epochs):
            # 运行一个周期
            mols, loss = self.run_episode(epoch)
            
            # 获取当前超体积
            hv = self.controller.compute_hypervolume()
            
            # ---- 记录权重历史 ----
            if self.controller._weights_history:
                current_weights = self.controller._weights_history[-1]
            else:
                current_weights = np.ones(self.num_obj) / self.num_obj
            self._weight_history.append((epoch, current_weights.copy(), hv))
            
            # ---- Cosine Annealing LR ----
            if self.config["use_cosine_lr"] and hasattr(self.agent, 'optimizer'):
                lr = self._get_cosine_lr(epoch)
                for param_group in self.agent.optimizer.param_groups:
                    param_group['lr'] = lr
            
            # 每10个epoch保存一次Pareto前沿数据 (使用配置)
            if epoch % self.config["checkpoint_freq"] == 0 or epoch == self.total_epochs - 1:
                self._save_pareto_checkpoint(epoch, hv, log_dir)
            
            # ---- 每轮都打印进度 ----
            phase = self.controller.get_phase(epoch)
            front_size = len(self.controller.pareto_front.solutions)
            n_valid = len(mols)
            
            # 进度条 + ETA
            progress = (epoch + 1) / self.total_epochs
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (self.total_epochs - epoch - 1)  # 秒
            eta_str = f"{eta:.0f}s" if eta < 120 else f"{eta/60:.1f}min"
            
            # 单行汇总 (用 \r 覆盖) + 换行时 flush
            summary = (f"\r[{bar}] {epoch+1}/{self.total_epochs} ({progress*100:.0f}%) "
                       f"| {phase} | Front:{front_size} | Valid:{n_valid} "
                       f"| HV:{hv:.4f} | Loss:{loss:.4f} | ETA:{eta_str}")
            print(summary, end="", flush=True)
            
            # 每10轮额外换行打印详细日志
            if epoch % 10 == 0:
                print()  # 换行，保留上一行进度
                print(f"  Epoch {epoch:3d} | Phase {phase} | "
                      f"Front Size: {front_size:3d} | "
                      f"Hypervolume: {hv:.4f} | "
                      f"Loss: {loss:.4f} | "
                      f"Valid Mols: {n_valid}")
        
        print()  # 最终换行
        total_time = time.time() - t_start
        print(f"⏱️ 训练耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
        print("🎉 训练完成！")
        print(f"最终帕累托前沿包含 {len(self.controller.pareto_front.solutions)} 个解")
        
        # 训练完成后保存最终结果并生成可视化
        self._save_final_results(log_dir)

    def _save_pareto_checkpoint(self, epoch: int, hv: float, log_dir: Path):
        """
        保存Pareto前沿检查点到CSV，并绘制3D/2D可视化图
        
        Args:
            epoch: 当前训练轮次
            hv: 当前超体积值
            log_dir: 日志目录路径
        """
        pareto_solutions = self.controller.pareto_front.solutions
        pareto_molecules = self.controller.pareto_front.molecules
        if not pareto_solutions:
            return
        
        # 准备CSV数据
        csv_path = log_dir / f'pareto_front_epoch_{epoch:03d}.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['smiles', 'egfr', 'vegfr2', 'qed', 'hypervolume', 'epoch'])
            
            # 写入每个Pareto解
            for sol_scores, mol in zip(pareto_solutions, pareto_molecules):
                writer.writerow([
                    mol.smiles,
                    sol_scores[0],  # EGFR
                    sol_scores[1],  # VEGFR2
                    sol_scores[2],  # QED
                    hv,
                    epoch
                ])
        
        print(f"  💾 Pareto前沿已保存: {csv_path.name} ({len(pareto_solutions)} 个解)")
        
        # 绘制3D和2D可视化图 (每10个epoch)
        solutions_array = np.array(pareto_solutions)
        if len(solutions_array) >= 3:  # 至少3个解才能画有意义的图
            self._plot_pareto_visualizations(
                solutions_array, epoch, log_dir,
                hv=hv, n_solutions=len(pareto_solutions)
            )
    
    def _plot_pareto_visualizations(self, solutions: np.ndarray, epoch: int,
                                     log_dir: Path, hv: float = 0.0,
                                     n_solutions: int = 0):
        """绘制并保存Pareto前沿的3D和2D可视化图"""
        import sys as _sys
        _vis_dir = Path(__file__).resolve().parent.parent / 'scripts' / 'visualization'
        _sys.path.insert(0, str(_vis_dir))
        try:
            from quick_visualize import plot_pareto_front_3d, plot_pareto_front_2d_pairs
            save_3d = log_dir / f'pareto_front_3d_epoch_{epoch:03d}.png'
            save_2d = log_dir / f'pareto_front_2d_epoch_{epoch:03d}.png'
            obj_names = ['EGFR', 'VEGFR2', 'QED']
            title = f"Pareto Front Epoch {epoch} | HV={hv:.4f} | Solutions={n_solutions}"
            
            plot_pareto_front_3d(solutions, objective_names=obj_names,
                                 title=title, save_path=str(save_3d))
            plot_pareto_front_2d_pairs(solutions, objective_names=obj_names,
                                       title=title, save_dir=str(log_dir))
            # 重命名2D图以包含epoch (因为函数固定输出到 save_dir)
            default_2d = log_dir / 'pareto_front_2d_pairs.png'
            if default_2d.exists():
                default_2d.replace(save_2d)
            print(f"  📊 可视化已保存: {save_3d.name}, {save_2d.name}")
        except ImportError:
            pass  # 可视化模块不可用时静默跳过

    def _save_final_results(self, log_dir: Path):
        """
        保存最终训练结果并生成可视化图表
        
        Args:
            log_dir: 日志目录路径
        """
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        
        # 尝试导入可视化函数 (从 scripts/visualization/ 目录)
        _vis_dir = Path(__file__).resolve().parent.parent / 'scripts' / 'visualization'
        sys.path.insert(0, str(_vis_dir))
        try:
            from quick_visualize import plot_pareto_front_3d, plot_pareto_front_2d_pairs
            visualization_available = True
        except ImportError:
            print("  ⚠️ 警告: 无法导入可视化模块，跳过图表生成")
            visualization_available = False
        
        pareto_solutions = self.controller.pareto_front.solutions
        pareto_molecules = self.controller.pareto_front.molecules
        
        if not pareto_solutions:
            print("  ⚠️ 警告: Pareto前沿为空，跳过可视化")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 最终帕累托前沿分析")
        print(f"{'='*80}")
        print(f"总帕累托解数量: {len(pareto_solutions)}")
        
        # 转换为numpy数组 (pareto_solutions已经是分数数组)
        final_solutions = np.array(pareto_solutions)
        
        # 确保至少有10个帕累托解用于输出
        target_num_solutions = min(10, len(pareto_solutions))
        print(f"选择前 {target_num_solutions} 个最优帕累托解进行详细展示")
        
        # 按照多目标综合评分排序（这里使用简单的加权和）
        # 也可以使用超体积贡献度排序
        if target_num_solutions < len(pareto_solutions):
            # 计算每个解的综合得分（归一化后的加权和）
            normalized_solutions = (final_solutions - final_solutions.min(axis=0)) / (final_solutions.max(axis=0) - final_solutions.min(axis=0) + 1e-8)
            composite_scores = normalized_solutions.sum(axis=1)
            # 选择综合得分最高的解
            top_indices = np.argsort(composite_scores)[-target_num_solutions:]
            top_solutions = [pareto_solutions[i] for i in top_indices]
            top_molecules = [pareto_molecules[i] for i in top_indices]
        else:
            top_solutions = pareto_solutions
            top_molecules = pareto_molecules
        
        # 保存前10个帕累托解到专门的文件
        self._save_top_10_pareto_solutions(top_solutions, top_molecules, log_dir)
        
        # 绘制3D图和2D配对图 (仅在可视化模块可用时)
        # ---- 保存权重历史 ----
        self._save_weight_history(log_dir)
        
        if visualization_available:
            save_path_3d = log_dir / 'pareto_front_3d_final.png'
            plot_pareto_front_3d(
                solutions=final_solutions,
                objective_names=['EGFR', 'VEGFR2', 'QED'],
                title=f"Pareto Front (Final) - {len(pareto_solutions)} Solutions",
                save_path=str(save_path_3d)
            )
            
            plot_pareto_front_2d_pairs(
                solutions=final_solutions,
                objective_names=['EGFR', 'VEGFR2', 'QED'],
                title=f"Pareto Front 2D Pairs (Final)",
                save_dir=str(log_dir)
            )
            print(f"  ✓ 可视化图表已保存到 logs/ 目录")

        # ---- 生成训练趋势折线图 (三个性质均值 + 超体积) ----
        self._plot_training_trend(log_dir)

        # ---- 生成三阶段权重轨迹双面板图 ----
        weight_csv = log_dir / 'weight_history.csv'
        if weight_csv.exists():
            try:
                parent_dir = Path(__file__).resolve().parent.parent
                sys.path.insert(0, str(parent_dir / 'scripts'))
                from plot_weight_trajectory import plot_weight_trajectory
                weight_plot_path = log_dir / 'weight_trajectory_three_stage.png'
                plot_weight_trajectory(
                    csv_path=str(weight_csv),
                    total_epochs=self.total_epochs,
                    output_path=str(weight_plot_path),
                )
                print(f"  📊 三阶段权重轨迹图已保存: {weight_plot_path.name}")
            except Exception as e:
                print(f"  ⚠️ 权重轨迹图生成失败: {e}")

        print(f"{'='*80}\n")

    def _plot_training_trend(self, log_dir: Path):
        """训练结束时，从所有 *_epoch_*.csv 聚合趋势并生成折线图"""
        import sys as _sys
        _script_dir = Path(__file__).resolve().parent.parent / 'scripts'
        _sys.path.insert(0, str(_script_dir))
        from plot_pareto_trend import collect_trend_data, plot_trends
        trend_df = collect_trend_data(str(log_dir))
        trend_path = log_dir / 'pareto_trend.png'
        plot_trends(trend_df, str(trend_path))
        print(f"  📈 训练趋势折线图已保存到: {trend_path.name}")

    def _save_weight_history(self, log_dir: Path):
        """保存权重历史到 CSV"""
        if not self._weight_history:
            return
        csv_path = log_dir / 'weight_history.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'w_egfr', 'w_vegfr2', 'w_qed', 'hypervolume'])
            for epoch, weights, hv in self._weight_history:
                writer.writerow([epoch, weights[0], weights[1], weights[2], hv])
        print(f"  📊 权重历史已保存到: {csv_path.name} ({len(self._weight_history)} 条记录)")
    
    def _save_top_10_pareto_solutions(self, solutions: List[np.ndarray], 
                                      molecules: List[Molecule], log_dir: Path):
        """
        保存前10个帕累托前沿解的详细信息
        
        Args:
            solutions: 帕累托解列表
            molecules: 对应的分子列表
            log_dir: 日志目录路径
        """
        output_path = log_dir / 'top_10_pareto_solutions.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 🏆 Top 10 帕累托前沿最优分子\n\n")
            f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"总帕累托解数量: {len(solutions)}\n")
            f.write(f"展示最优解数量: {len(solutions)}\n\n")
            f.write("---\n\n")
            
            # 计算每个解的综合得分
            solutions_array = np.array(solutions)
            normalized_solutions = (solutions_array - solutions_array.min(axis=0)) / (solutions_array.max(axis=0) - solutions_array.min(axis=0) + 1e-8)
            composite_scores = normalized_solutions.sum(axis=1)
            
            # 按综合得分排序
            sorted_indices = np.argsort(composite_scores)[::-1]
            
            for rank, idx in enumerate(sorted_indices, 1):
                mol = molecules[idx]
                scores = solutions[idx]
                
                f.write(f"## 🥇 排名 #{rank}\n\n")
                f.write(f"### 分子信息\n")
                f.write(f"**SMILES**: `{mol.smiles}`\n\n")
                f.write(f"### 目标得分\n")
                f.write(f"- **EGFR亲和力**: {scores[0]:.4f}\n")
                f.write(f"- **VEGFR2亲和力**: {scores[1]:.4f}\n")
                f.write(f"- **QED (药物相似性)**: {scores[2]:.4f}\n")
                f.write(f"- **综合得分**: {composite_scores[idx]:.4f}\n\n")
                
                # 计算额外的分子属性
                from rdkit.Chem import Descriptors
                rdkit_mol = Chem.MolFromSmiles(mol.smiles)
                if rdkit_mol is not None:
                    f.write(f"### 分子属性\n")
                    f.write(f"- **分子量**: {Descriptors.MolWt(rdkit_mol):.2f} Da\n")
                    f.write(f"- **LogP**: {Descriptors.MolLogP(rdkit_mol):.2f}\n")
                    f.write(f"- **TPSA**: {Descriptors.TPSA(rdkit_mol):.2f} Å²\n")
                    f.write(f"- **H-键供体**: {Descriptors.NumHDonors(rdkit_mol)}\n")
                    f.write(f"- **H-键受体**: {Descriptors.NumHAcceptors(rdkit_mol)}\n")
                    f.write(f"- **可旋转键**: {Descriptors.NumRotatableBonds(rdkit_mol)}\n\n")
                
                f.write("---\n\n")
        
        print(f"  📄 前{len(solutions)}个帕累托解已保存到: {output_path.name}")
        
        # 同时保存为CSV格式便于分析
        csv_path = log_dir / 'top_10_pareto_solutions.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['rank', 'smiles', 'egfr', 'vegfr2', 'qed', 'composite_score'])
            for rank, idx in enumerate(sorted_indices, 1):
                mol = molecules[idx]
                scores = solutions[idx]
                writer.writerow([
                    rank,
                    mol.smiles,
                    f"{scores[0]:.6f}",
                    f"{scores[1]:.6f}",
                    f"{scores[2]:.6f}",
                    f"{composite_scores[idx]:.6f}"
                ])
        
        print(f"  📊 CSV格式数据已保存到: {csv_path.name}")
# ==================== 启动程序 ====================
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 创建集成器 - 使用新训练的 Polygon VAE 模型
    base_dir = Path(__file__).resolve().parent.parent
    # 更优质的 Polygon VAE (3层解码器，内置词汇表，无需外部vocab)
    default_model = base_dir / 'models' / 'polygon_v2_strongdec.pt'
    
    # 如果模型不存在，回退到旧版模型 (需要vocab)
    if not default_model.exists():
        default_model = base_dir / 'models' / 'polygon_anneal_080.pt'
    if not default_model.exists():
        default_model = base_dir / 'models' / '复现通用训练' / 'trained_vae.pt'
    
    # Polygon VAE 内置词汇表，vocab_path 仅用于 Legacy 回退
    default_vocab = base_dir / 'vocab' / 'twoG_smiles.pkl'
    vocab_kwarg = {"vocab_path": str(default_vocab)} if not default_model.name.startswith('polygon') else {}
    
    default_egfr = base_dir / 'models' / '蛋白靶点预测器' / 'target_EGFR_model.pkl'
    default_vegfr2 = base_dir / 'models' / '蛋白靶点预测器' / 'target_VEGFR2_model.pkl'
    
    print(f"使用VAE模型: {default_model}")
    if vocab_kwarg:
        print(f"使用词表: {default_vocab}")
    
    # 支持命令行参数覆盖模型路径、温度和epochs
    # 用法: python main_pipeline.py [model_path] [temperature] [epochs]
    # 例如: python main_pipeline.py models/polygon_v2_strongdec.pt 1.0 500
    model_path = str(default_model)
    temperature = 0.7
    total_epochs = 1000
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        temperature = float(sys.argv[2])
    if len(sys.argv) > 3:
        total_epochs = int(sys.argv[3])
    
    print(f"使用VAE模型: {model_path}")
    print(f"采样温度: {temperature}")
    print(f"总训练轮次: {total_epochs}")
    if vocab_kwarg:
        print(f"使用词表: {default_vocab}")
    
    integrator = MO_RL_Integrator(
        latent_dim=128,
        num_obj=3,
        total_epochs=total_epochs,
        batch_size=32,
        vae_model_path=model_path,
        **vocab_kwarg,
        egfr_model_path=str(default_egfr),
        vegfr2_model_path=str(default_vegfr2),
        config={"temperature": temperature}
    )
    
    # 开始训练
    integrator.train()
