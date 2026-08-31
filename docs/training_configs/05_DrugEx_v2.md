# DrugEx v2训练与Evolve优化配置

## 环境与源码

- Python 3.10.20。
- 环境：`environments/requirements_snapshot_drugex_v2.txt`。
- 上游revision：`abcc5091bd526c639ac3a0b6128963537652fb0b`。

## 数据编码

入口：`baselines/drugex_v2_adapter/adapter_prepare_common_data.py`。

| 参数 | 值 |
|---|---:|
| 输入 | `data/train_smiles_only.txt` |
| Max model tokens | 100 |
| 实际最大tokens | 54 |
| 接受/拒绝 | 1,584,663 / 0 |
| Vocabulary | 24个数据token；含控制token共26 |
| 输出 | `common_tokens.npy`、`common_voc.txt` |

这些编码文件属于可再生缓存，不需要上传。

## 底层生成器训练

入口：`baselines/drugex_v2_adapter/adapter_train_common_prior.py`。

| 参数 | 值 |
|---|---:|
| 初始化 | 随机 |
| 架构 | DrugEx v2原生LSTM Generator |
| Epochs上限 | 20 |
| Batch size | 512 |
| Adam learning rate | 0.001 |
| Workers | 8 |
| Seed | 42 |
| Train/validation | 按编码数据顺序前95% / 后5% |
| Early stopping patience | 3 |
| Mixed precision | CUDA bfloat16 autocast |
| Gradient clip | 5.0 |
| 验证上限 | 128 batches/epoch |
| 选模 | 最低validation loss |

## 正式Evolve/Pareto-ranking优化

| 参数 | 值 |
|---|---:|
| Budget | 10,240 |
| Seeds | 42--51 |
| Batch size | 64 |
| Replay batches/update | 10 |
| Oracle calls/update | 640 |
| Policy updates | 16 |
| Epsilon | 0.001 |
| Reward scheme | PR（Pareto-ranking相似性奖励） |
| PGLoss loader batch | 128 |
| Agent/prior/crossover初始化 | 共同数据集最佳先验 |
| Prior/crossover | 冻结 |

## 执行入口

```powershell
pwsh ./train_baseline_bottom_model.ps1 -Method drugex_v2 -RebuildPreprocessedData -Workers 8
pwsh ./run_baseline_generation.ps1 -Method drugex_v2 -TargetPair EGFR_VEGFR2 -Budget 10240 -Seed 42 -Evaluate
```

## 参数证据

- `results/baselines/drugex_v2/data/metadata.json`
- `results/baselines/drugex_v2/formal_10240_seed42/metadata.json`
- `baselines/drugex_v2_adapter/adapter_prepare_common_data.py`
- `baselines/drugex_v2_adapter/adapter_train_common_prior.py`
- `baselines/drugex_v2_adapter/adapter_optimize_dual_oracle.py`

