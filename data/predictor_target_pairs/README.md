# 双靶点预测器数据归档

整理日期：2026-08-30

本目录只保存两组靶点预测器直接相关、且后续可用于审计或复现的数据。原始文件均保留在原位置，本目录中的文件为带明确标识的副本。

## 01_EGFR_VEGFR2_第一组_恢复相关数据_NOT_EXACT_ORIGINAL

第一组靶点：EGFR（UniProt P00533）与 VEGFR2（UniProt P35968）。

重要边界：目前没有找到训练现存 POLYGON 随机森林预测器时所用的精确 `BindingDB_All.csv/tsv` 快照，因此本目录不能标记为“原始训练集”。

- `EGFR_P00533_BindingDB_API_snapshot_20260712.json`：BindingDB API 快照，32,197 条原始 affinity 记录。
- `VEGFR2_P35968_BindingDB_API_snapshot_20260712.json`：BindingDB API 快照，15,902 条原始 affinity 记录。
- `POLYGON_original_RF_training_script.py`：原随机森林训练流程，用于说明 BindingDB 筛选和 ECFP4/RF1000 训练方式。

POLYGON论文源数据工作簿不参与本项目预测器训练，已从精简包移出。

现存模型内部推断训练规模为 EGFR 1,096 条、VEGFR2 723 条；模型文件不保存原始 SMILES 训练行。

## 02_PARP1_BRD4_第二组_当前正式预测器数据_ChEMBL37

第二组靶点：PARP1（ChEMBL CHEMBL3105）与 BRD4（ChEMBL CHEMBL1163125）。数据版本为 ChEMBL 37。

当前正式“EGFR/VEGFR2 参数对齐”预测器实际使用：人源单蛋白 binding assay、精确关系、IC50、nM；ECFP4 radius=2、2048 bits、包含手性；RandomForestRegressor 1000 棵树、max_features=1.0、min_samples_leaf=1、random_state=0。

- `PARP1_CHEMBL3105_train_through_2023_n2538.csv`：当前 PARP1 正式模型训练集。
- `PARP1_CHEMBL3105_audit_2024plus_n192.csv`：PARP1 2024+ 重复审计集，不属于训练集。
- `BRD4_CHEMBL1163125_train_through_2023_n5245.csv`：当前 BRD4 正式模型训练集。
- `BRD4_CHEMBL1163125_audit_2024plus_n333.csv`：BRD4 2024+ 重复审计集，不属于训练集。
- `CURRENT_ALIGNED_PREDICTOR_METADATA.json`：当前正式对齐预测器的参数、用途边界与数据哈希。
- `DATA_MANIFEST.json`、`RAW_DOWNLOAD_MANIFEST.json`：数据筛选、拆分与下载来源记录。
- `TRAINING_REPORT_ZH.md`：当前正式对齐预测器训练报告。

两份ChEMBL原始下载JSONL属于可追溯但非训练必需的中间快照，精简包只保留固定训练/审计CSV与下载清单。

## 使用原则

1. 第一组数据只能称为“恢复的相关数据/可重建数据”，不能称为精确原始训练集。
2. 第二组 `train_through_2023` 文件是当前正式预测器的训练数据；`audit_2024plus` 仅用于审计。
3. 论文中的绝对活性结论仍需实验验证；这些预测器适合共享奖励、相对排序和方法间统一比较。
