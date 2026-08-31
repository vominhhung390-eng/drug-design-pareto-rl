# CLOVER-Mol V4-B训练与生成配置

## 运行身份

| 项目 | 锁定值 |
|---|---|
| 版本 | `V4-B raw-mean` |
| Runner | `method/ablation/run_wc_two_targets_multiexplore.py` |
| 正式比较种子 | 42--51 |
| 前瞻确认种子 | 82--91（不替代主表的共同种子） |
| 每种子预算 | 10,240 |
| Batch | 64 |
| Epochs | 160（10,240 / 64） |
| 设备 | CUDA |
| 轨迹长度 | 1 |
| 轨迹步长归一化 | `sqrt` |
| Controller | `ours_full_corrected` |
| Critic | `multi` |
| 权重模式 | `dynamic` |
| 探索模式 | `multiscale` |
| 通道模式 | `adaptive` |

## PPO参数

| 参数 | 值 |
|---|---:|
| 学习率 | 0.0003 |
| PPO epochs | 4 |
| Mini-batch size | 32 |
| Entropy coefficient | 0.01 |
| Value-loss coefficient | 0.5 |
| Base step scale | 0.08 |
| Latent clip | 4.0 |
| Invalid reward | -1.0 |
| Dirichlet alpha | 0.5 |
| Preference floor | 0.1 |
| Preference EMA alpha | 0.25 |

## Archive与动态偏好参数

| 参数 | 值 |
|---|---:|
| `archive_seed_fraction` | 0.0 |
| `archive_seed_noise` | 0.10 |
| `archive_seed_noise_end` | 0.05 |
| `archive_seed_start` | 0.30 |
| `archive_seed_ramp_end` | 0.70 |
| `archive_seed_selection` | `uniform` |
| `archive_hvc_weight` | 0.7 |
| `archive_balance_weight` | 0.3 |
| `archive_selection_temperature` | 0.25 |
| `archive_uniform_mix` | 0.10 |
| `archive_stagnation_window` | 0 |
| `archive_stagnation_delta` | 0.002 |
| `archive_stagnation_noise` | 0.0 |
| `sample_preference_mode` | `shared` |
| `sample_preference_blend` | 0.0 |
| `sample_preference_start` | 0.30 |
| `sample_preference_ramp_end` | 0.70 |
| `pareto_reward_start` | 0.30 |
| `pareto_reward_ramp_end` | 0.70 |

正式V4-B中 `hvc_reward_weight`、`crowding_reward_weight`、`balanced_reward_weight` 和 `pareto_actor_coef` 均为0.0。

## 在线生成器微调

| 参数 | 值 |
|---|---:|
| Actor mode | `train` |
| 微调间隔 | 16个batch，即1,024次oracle调用 |
| 每次微调epochs | 2 |
| Elite数量 | 512 |
| Elite策略 | `raw_mean` |
| 微调batch | 512 |
| 微调学习率 | 0.0003 |

## 输入和输出

- 共享VAE应由 `03_POLYGON.md` 的预训练阶段生成。
- 四个预测器应由 `08_四靶点RF预测器.md` 生成。
- Novelty训练缓存必须由 `data/train_smiles_only.txt` 确定性生成；不能把旧缓存作为必需下载资产。
- 每个种子的输出必须保存 `resolved_config.json`、生成分子CSV、anytime评价和质量约束评价。

## 两个靶点对

EGFR/VEGFR2和PARP1/BRD4使用完全相同的V4-B优化参数。第二靶点对只替换预测器，内部兼容列 `egfr/vegfr2` 分别映射为PARP1/BRD4。

## 参数证据

- `results/own_method_v4/common_seeds_42_51_10240/v4_b_raw_mean_seed42/resolved_config.json`
- `config/selected_own_method.json`
- `config/v4_raw_mean_common_seeds_42_51_10240.json`
- `run_target_pair_own_method_worker.ps1`

