# MO-LSO训练与GPflow优化配置

## 环境与源码

- 上游版本：GMD-MO-LSO v1.0.0。
- Torch/JTVAE环境：Python 3.10.20，`environments/requirements_snapshot_mo_lso_torch.txt`。
- GPflow环境：Python 3.10.20，`environments/requirements_snapshot_mo_lso_gpflow.txt`。
- Torch与GPflow必须隔离；GPflow子进程强制CPU。

## JTVAE数据预处理

入口：`baselines/mo_lso_adapter/adapter_prepare_common_data.py`。

| 参数 | 正式记录值 |
|---|---:|
| 输入 | `data/train_smiles_only.txt` |
| Workers | 24 |
| Shard size | 5,000 |
| Shards | 317 |
| 接受分子 | 1,497,221 |
| 拒绝分子 | 87,442 |
| Vocabulary size | 484 |

预处理张量约4.14 GB，属于可再生缓存，不上传。

## JTVAE底层模型训练

入口：`baselines/mo_lso_adapter/adapter_train_common_jtvae.py`。

| 参数 | 值 | 状态 |
|---|---:|---|
| Hidden size | 450 | 已验证 |
| Latent size | 56 | 已验证 |
| Tree latent size | null | 已验证 |
| Tree depth | 20 | 已验证 |
| Graph depth | 3 | 已验证 |
| 初始化 | 官方wrapper一致的Xavier/zero随机初始化 | 已验证 |
| Epochs上限 | 30 | 启动脚本锁定 |
| Batch size | 32 | 已验证 |
| Learning rate | 0.0007 | 已验证 |
| Beta | 0.005 | 已验证 |
| Gradient clip | 20.0 | 代码默认且启动未覆盖 |
| Validation fraction | 0.05（按shard末尾切分） | 代码默认 |
| Train/validation shards | 301 / 16 | 已验证 |
| Patience | 3 | 启动脚本锁定 |
| Seed | 42 | 已验证 |
| Device | `cuda:0` | 已验证 |
| 选模 | 最低validation loss | 已验证 |

训练worker数未写入正式 `training_config.json`；当前统一入口默认8。发布时应在命令中显式固定该值。

## 正式MO-LSO优化

| 参数 | 值 | 状态 |
|---|---:|---|
| Budget | 10,240 | 已验证 |
| Seeds | 42--51 | 已验证 |
| Batch size | 64 | 已验证 |
| Candidate pool | 4,096 | 代码默认 |
| Retraining frequency | 1,024 oracle calls | 已验证 |
| Retraining learning rate | 0.0001 | 代码默认 |
| Rank-weight k | 0.001 | 已验证 |
| Beta | 0.005 | 代码默认 |
| GP warmup | 256 | 已验证 |
| Inducing points | 128 | 已验证 |
| GP starts | 128 | 已验证 |
| GP workers | 1 | 已验证 |
| GP optimization iterations | 100,000 | 已验证 |
| Acquisition | 官方GPflow SGPR + Expected Improvement on Pareto rank | 已验证 |
| 目标适配 | 两个归一化oracle的Pareto rank，均最大化 | 已验证 |

## 执行入口

```powershell
pwsh ./train_baseline_bottom_model.ps1 -Method mo_lso -RebuildPreprocessedData -Workers 8
pwsh ./run_baseline_generation.ps1 -Method mo_lso -TargetPair EGFR_VEGFR2 -Budget 10240 -Seed 42 -Evaluate
```

## 参数证据

- `results/baselines/mo_lso/data/metadata.json`
- `results/baselines/mo_lso/models/training_config.json`
- `results/baselines/mo_lso/formal_10240_seed42/metadata.json`
- `baselines/mo_lso_adapter/adapter_*`

