# 已知复现边界

1. **EGFR/VEGFR2 精确历史训练行缺失，但历史模型已保留。** 默认正式流程会校验并加载 `models/oracles/` 中的两个历史RF，因此可复现论文使用的固定oracle输出；但不能从最初BindingDB行级数据重新训练出同一模型。2026-07-12恢复的BindingDB API快照属于另一数据条件，只有显式传入 `--allow-recovered-egfr-vegfr2` / `-AllowRecoveredEgfrVegfr2` 才会用于替代重训。
2. **POLYGON 是共享增强VAE版本。** 其底层优化机制仍为POLYGON，但生成器用 canonical + 3 randomized SMILES 数据训练，与CLOVER-Mol共享同一checkpoint。这对基线有利，但不等于无增强的严格原版POLYGON。
3. **随机性与库版本。** CUDA、PyTorch、RDKit和scikit-learn版本必须记录。重新训练模型哈希可能因底层非确定性不同，评价应以固定参数、相同数据、种子和oracle预算为准。
4. **预测结果不是实验活性。** 四个RF仅用于所有方法共享的相对排序与优化奖励，不能据此宣称真实生物活性；Vina分数也不是实验结合证据。
5. **顶层发布许可证待作者选择。** 上游仓库许可证均保留，但CLOVER-Mol自身尚无顶层 `LICENSE`，`CITATION.cff` 也缺少作者/单位/DOI信息；公开GitHub发布前需补。
6. **Vina 1.1.2二进制需用户自行提供。** 公共仓库保留全部受体、box、准备/运行/汇总脚本和2,400条参考任务，但不再分发旧版Windows可执行文件。运行时设置 `VINA_EXECUTABLE` 或将二进制放入约定路径；改用新版必须记录版本差异。

若以后获得最初 EGFR/VEGFR2 BindingDB TSV，应放入第一靶点对数据目录，并在训练脚本中增加精确数据分支、记录SHA-256后，才能补齐“训练数据级复现”；在此之前只能承诺“固定模型输出级复现”。
