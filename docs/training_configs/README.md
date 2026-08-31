# CLOVER-Mol 完整复现训练配置索引

本目录按训练或实验环节分别记录配置。所有参数分为三种状态：

- **已验证**：可由正式运行的 `resolved_config.json`、`metadata.json`、训练配置或启动脚本直接确认。
- **代码默认**：当前适配器在未显式覆盖时使用的默认值；可复跑，但不能等同于已经证明的历史命令行。
- **待恢复**：当前文件中没有足够证据，发布前不能自行补写为原始参数。

| 编号 | 环节 | 配置文件 | 当前状态 |
|---|---|---|---|
| 01 | 共同数据、预算、种子、靶点和资源协议 | [01_共同实验协议.md](01_共同实验协议.md) | 已验证 |
| 02 | CLOVER-Mol V4-B训练与生成 | [02_CLOVER-Mol_V4-B.md](02_CLOVER-Mol_V4-B.md) | 已验证 |
| 03 | 共享增强POLYGON VAE预训练与POLYGON正式优化 | [03_POLYGON.md](03_POLYGON.md) | 完整历史参数、增强、选模规则均已恢复 |
| 04 | REINVENT4底层模型训练与DAP优化 | [04_REINVENT4.md](04_REINVENT4.md) | 已验证 |
| 05 | DrugEx v2数据编码、先验训练与Evolve优化 | [05_DrugEx_v2.md](05_DrugEx_v2.md) | 已验证 |
| 06 | MO-LSO预处理、JTVAE训练与GPflow优化 | [06_MO-LSO.md](06_MO-LSO.md) | 大部分已验证 |
| 07 | GraphPareto--NSGA-II直接搜索 | [07_GraphPareto-NSGA-II.md](07_GraphPareto-NSGA-II.md) | 已验证；无底层训练阶段 |
| 08 | EGFR、VEGFR2、PARP1、BRD4四个RF预测器 | [08_四靶点RF预测器.md](08_四靶点RF预测器.md) | 四靶点训练参数已统一；第一靶点对数据非精确原始版 |
| 09 | 评价指标、10种子汇总与论文表格 | [09_评价与论文表格.md](09_评价与论文表格.md) | 已验证；统一入口已补 |
| 10 | 每种子独立Top10分子对接 | [10_Top10分子对接.md](10_Top10分子对接.md) | 已验证；1,200个候选、2,400项对接 |

## 复现顺序

1. 按 `environments/` 重建各方法环境。
2. 按 `08_四靶点RF预测器.md` 训练四个共享预测器。
3. 按 `03_POLYGON.md` 训练共享POLYGON VAE。
4. 分别训练 REINVENT4、DrugEx v2 和 MO-LSO 的底层生成模型；GraphPareto无需训练。
5. 用相同数据集、预测器、预算和种子运行 CLOVER-Mol 与五个基线。
6. 计算标准指标和质量约束指标，按种子汇总论文表格。
7. 对每个靶点对、方法和种子分别在种子内去重并独立选Top10，然后完成分子对接；不得跨种子合并排名或去重。

## 发布前阻断项

以下问题未解决前，只能称为“从头方法复现”，不能称为“数值完全重建原论文模型”：

1. 四个RF预测器的正式特征与训练参数已经确认统一；但EGFR/VEGFR2目录为恢复相关数据，不是已通过来源和哈希确认的最初训练数据。
2. `run_predictor_retraining.ps1` 是后续Chemprop实验，不是论文四个正式RF预测器的训练入口；正式入口是 `scripts/train_four_rf_predictors.py`。
3. POLYGON必须在论文中标记为使用共享增强VAE，不能称为无增强的严格原版。
