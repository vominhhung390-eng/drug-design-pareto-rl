# REINVENT4训练与DAP优化配置

## 环境与源码

- Python 3.11.15。
- 依赖锁：`baselines/reinvent4_adapter/pyproject.toml` 和 `uv.lock`。
- 当前来源记录为 `REINVENT4-main` archive；正式发布前应补充archive SHA-256或上游commit。

## 随机初始化先验

配置：`baselines/reinvent4_adapter/local_configs/create_common_dataset_prior.toml`。

| 网络参数 | 值 |
|---|---:|
| Layers | 3 |
| Layer size | 512 |
| Cell | LSTM |
| Embedding | 256 |
| Dropout | 0.0 |
| Max sequence length | 128 |
| Layer normalization | false |
| Standardize | false |
| 初始化 | 随机初始化，不允许预训练权重 |

## 共同数据集训练

配置：`baselines/reinvent4_adapter/local_configs/train_common_dataset_prior.toml`。

| 参数 | 值 |
|---|---:|
| Run type | transfer learning |
| Device | `cuda:0` |
| Epochs | 10 |
| Batch size | 512 |
| Save frequency | 每1 epoch |
| Number of references | 0 |
| Sample batch size | 1,024 |
| TensorBoard internal similarity | false |
| Standardize SMILES | false |
| Seed | 42 |
| 输入 | 随机初始化模型＋`train_smiles_only.txt` |
| 输出 | `reinvent4_common_dataset.model` |

## 正式DAP优化

| 参数 | 值 |
|---|---:|
| Budget | 10,240 |
| Seeds | 42--51 |
| Batch size | 64 |
| Iterations | 160 |
| Algorithm | DAP |
| 双目标聚合 | 几何均值 |
| Sigma | 128.0 |
| Adam learning rate | 0.0001 |
| Gradient clip | 5.0 |
| Prior | 冻结的、由共同数据从头训练的prior |
| Agent initialization | 与prior相同 |

## 执行入口

```powershell
pwsh ./train_baseline_bottom_model.ps1 -Method reinvent4
pwsh ./run_baseline_generation.ps1 -Method reinvent4 -TargetPair EGFR_VEGFR2 -Budget 10240 -Seed 42 -Evaluate
```

## 参数证据

- `baselines/reinvent4_adapter/local_configs/*.toml`
- `baselines/reinvent4_adapter/adapter_optimize_dual_oracle.py`
- `results/baselines/reinvent4/formal_10240_seed42/metadata.json`
- `train_baseline_bottom_model.ps1`

