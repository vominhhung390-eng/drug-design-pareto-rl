# 精简说明

2026-08-31 将以下可再生或历史资产移出公开发布包；作者的本地归档仍保留原始副本，没有直接删除：

- `results/`：已完成正式结果、MO-LSO预处理张量、底层模型和日志；小型最终主表另存于 `reference_results/paper_tables/`。
- `models/`：共享VAE与四个RF权重；从训练参数和数据重新生成。
- `outputs/` 与 `data/train_canonical_cache.txt`：Novelty canonical缓存；由表格脚本重新生成。
- `docking/seed_top10_two_pairs_20260830/`：配体、poses与原始对接结果；小型CSV/JSON汇总另存于 `reference_results/docking_top10/`。
- `baselines/mo_lso_adapter/assets/pretrained_models/`：不用于本论文“从随机初始化训练”的上游预训练权重。
- `data/external/`：与最终两组预测器数据重复或属于已退役靶点的下载快照。
- 两份ChEMBL原始下载JSONL和不参与训练的POLYGON论文源XLSX：正式训练/审计CSV、恢复的BindingDB JSON及下载清单已经足够完成论文复现。
- DrugEx上游 `LIGAND_RAW.tsv`、POLYGON旧甲基化编码、MO-LSO上游预训练数据：均不进入本论文共同数据集从头训练流程。
- PARP1/BRD4受体制备过程的超大JSON：Vina复现只需要保留的原始PDB、正式PDBQT、晶体配体、box和验证结果。
- GraphPareto上游 `nsga-iii/`、`nsga-ii/data/models/` 和 `nsga-ii/data/smiles/`：不用于本论文NSGA-II双预测器实验；`nsga-ii/data/smarts/`完整保留。

发布包保留全部正式数据、源码、配置、环境快照、受体、训练/生成/评价/对接入口和小型参考结果。历史Vina 1.1.2可执行文件按第三方许可边界由用户自行提供；运行 `scripts/reproduce_all.ps1` 会在原路径重建其他移出的内容。
