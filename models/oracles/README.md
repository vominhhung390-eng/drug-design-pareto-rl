# 历史 EGFR / VEGFR2 正式预测器

本目录保存论文正式生成和评价所使用的历史训练模型。由于最初的
BindingDB 行级训练表已经无法恢复，默认复现流程直接加载这些固定模型，
以复现历史 oracle 的实际输出，而不是用不同数据重新拟合后冒充同一预测器。

| 模型 | 文件 | SHA-256 | 历史训练样本数（由随机森林根节点权重推断） |
|---|---|---|---:|
| EGFR | `target_EGFR_model.pkl` | `d57ba46d71c7a943c3e17a6a6a688d55d48d9cbed93fa1429d95dedb85ae03ab` | 1,096 |
| VEGFR2 | `target_VEGFR2_model.pkl` | `c2cfd492cfc0fa367dd8a262f5716b7eea4d1c2de3ff25acccdd96bae9eeeb94` | 723 |

两者均为 scikit-learn `RandomForestRegressor`，包含1,000棵树，
`max_features=1.0`、`min_samples_leaf=1`、`random_state=0`，输入维度为2,048。
项目正式打分接口使用 radius=2、2,048-bit、启用手性的 Morgan 指纹。

复现边界：这些文件支持“固定模型输出级复现”，但因为缺少精确训练行，
不支持“从原始行级数据重新训练并得到同一模型”的复现。包内恢复的 BindingDB
快照仅用于显式选择的替代重训实验，不能与本目录模型混称为同一条件。
