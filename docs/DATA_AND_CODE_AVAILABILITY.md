# 数据与代码可用性说明

## 包内可直接复现的数据

- 共同生成训练集：`data/train_smiles_only.txt`，SHA-256见根目录关键哈希表。
- PARP1/BRD4：ChEMBL 37正式训练CSV与2024+重复审计CSV。
- EGFR/VEGFR2：恢复的BindingDB API快照；不是精确历史训练行，一键预检要求显式同意。
- 分子对接：四个正式PDBQT受体、原始PDB/晶体配体、box配置、协议验证和Vina调用脚本。历史Vina 1.1.2可执行文件属于用户自行取得的第三方运行时，不在公共仓库中再分发。

所有派生缓存、模型权重和完整正式结果均由 `scripts/reproduce_all.ps1` 重建；`reference_results/` 仅作为输出核对参考，不是训练输入。

## 来源和用途边界

外部数据的靶点、版本、筛选条件与文件哈希记录在 `data/predictor_target_pairs/` 和 `SOURCE_MAP_来源映射.csv`。RF预测值只作为所有方法共享的相对排序oracle；对接分数只作为计算筛选结果，均不构成实验活性或临床有效性证据。

## 发布前仍需作者决定

五个上游方法和vendored POLYGON均保留各自LICENSE。CLOVER-Mol自身代码目前没有作者选定的顶层开源许可证，也没有作者姓名/单位/DOI所需信息，因此包内未擅自生成 `LICENSE` 或 `CITATION.cff`。公开GitHub发布前应由作者选择许可证并补充引用元数据。
