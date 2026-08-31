# 共享增强 POLYGON VAE 与 POLYGON 适配器

## 方法身份边界

本项目的 CLOVER-Mol 与 POLYGON 适配器共享同一个 VAE 生成起点。该 VAE 使用共同原始数据集的“canonical + 3条 randomized SMILES”增强版本训练。因此论文和表格中应写为 **POLYGON（shared augmented VAE）**，不能写成“无增强的严格原版 POLYGON”。这样做保持生成器起点公平，但会增强 POLYGON 基线；属于对基线有利的保守比较。

训练实现来自 `vendor/polygon-main/`，正式训练入口为：

```powershell
pwsh ./scripts/train_shared_polygon_vae.ps1
```

全部机器可读参数保存在 `config/reproduction_pipeline.json` 的 `shared_polygon_vae` 节点。

## 数据增强

| 参数 | 锁定值 |
|---|---:|
| 原始数据 | `data/train_smiles_only.txt` |
| 原始行数 | 1,584,663 |
| 原始 SHA-256 | `4301e7f6118839465012eb93510328681ef4b7b24642e8748c4ad40971f4a304` |
| 每分子表示 | canonical + 3条 randomized SMILES |
| 增强 seed | 20260701 |
| workers | 24 |
| chunk size | 8,000 |
| 历史输出行数 | 6,324,009 |
| 历史输出 SHA-256 | `6fdbbb4588aeef6e0d4f3b4bf8d3b834ea6d5af277dd12c2d918c1f24e466ea2` |

增强文件属于可再生缓存，不随精简包分发。

## VAE 正式训练

| 参数 | 锁定值 |
|---|---:|
| seed | 20260702 |
| validation data | 无 |
| epochs | 70 |
| batch size | 2,048 |
| save frequency | 每1 epoch |
| device | `cuda:0` |
| Encoder | GRU，双向，hidden 512，1层，dropout 0.30 |
| Decoder | GRU，3层，dropout 0.20，hidden 512 |
| Latent dimension | 128 |
| LR start / end | 0.00025 / 0.00025 |
| KL start epoch | 10 |
| KL weight start / end | 0.0 / 0.15 |
| Gradient clip | 50 |
| DataLoader workers | 0 |
| SGDR period / restarts / multiplier | 10 / 6 / 1 |
| Middle dropout / layers | 0.2 / 1 |
| BatchNorm conv / middle | true / true |
| Lambda scale | 1.0 |

早期文档中的 encoder dropout 0.5、epoch 200 和 batch 1024 是官方示例或代码默认值，不是本项目正式训练值，已废止。

## checkpoint 选模

| 参数 | 锁定值 |
|---|---:|
| 候选 | epoch 20、25、30、35、40、45、50、55、60、65 与 final |
| 每候选采样 | 20,000 |
| 评价 seed | 42 |
| batch | 512 |
| max SMILES length | 120 |
| 温度 | 1.00、1.05、1.10、1.15、1.18、1.20、1.22、1.25、1.30、1.35 |
| 排名 | validity≥0.90且novelty≥0.90；再依次最大化min(validity, novelty)、两者之和、uniqueness |
| 历史胜出 | epoch 20，temperature 1.00 |
| 历史模型 SHA-256 | `f26bc67c3d6ea47c5756941902a52b8607df170f24528f7df6aa1db59fc504e5` |

历史胜出点的20,000样本结果为 validity 0.94245、novelty 0.960522、uniqueness 0.9998408。重新训练允许因硬件/库非确定性产生数值差异，但必须使用同一参数和选模规则。

## POLYGON 正式优化

| 参数 | 锁定值 |
|---|---:|
| Budget | 10,240 |
| Seeds | 42--51 |
| Batch size | 1,024 |
| Iterations | 10 |
| Keep top | 512 |
| Finetune epochs | 2 |
| 命令行 finetune batch | 256 |
| Learning rate | 0.0003 |
| Temperature | 1.0 |
| Max SMILES length | 100 |
| 排名标量 | 两个归一化目标的均值 |

正式优化仍使用官方 POLYGON 训练/生成机制；项目适配层只负责统一双预测器、oracle 预算、输出和评价。
