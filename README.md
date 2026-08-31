# CLOVER-Mol 精简完整复现包

这个目录只保留从头复现论文实验所需的源码、数据、环境快照、训练参数、统一评价和分子对接脚本。历史权重、预处理张量、运行日志和大体积正式输出不作为下载依赖；它们都能由脚本重新生成。

## 从 GitHub 获取

共同训练集由 Git LFS 管理。首次下载前请安装 Git LFS，然后克隆仓库并确认大文件已经拉取：

```powershell
git lfs install
git clone https://github.com/vominhhung390-eng/drug-design-pareto-rl.git
cd drug-design-pareto-rl
git lfs pull
```

正常克隆后，`data/train_smiles_only.txt` 应为 1,584,663 行，SHA-256 为 `4301e7f6118839465012eb93510328681ef4b7b24642e8748c4ad40971f4a304`；一键预检也会自动核对。

历史对接使用 AutoDock Vina 1.1.2。其旧版 Windows 二进制不在公共仓库中再分发；运行对接前请自行取得该版本，并设置环境变量 `VINA_EXECUTABLE`，或放到 `tools/autodock_vina_1.1.2/vina.exe`。当前官方 Vina 版本和安装方式见 [AutoDock Vina 官方仓库](https://github.com/ccsb-scripps/AutoDock-Vina)；若改用新版，必须记录版本，且不能将结果表述为与历史1.1.2逐值一致。

## 一键入口

提供两个入口：

- `一键复现.bat`：严格入口。若缺少论文历史使用的 EGFR/VEGFR2 原始行级训练数据，会在预检阶段停止，不会静默替换数据。
- `一键复现_允许恢复EGFR_VEGFR2数据.bat`：可运行入口。明确接受包内已标注来源的恢复数据，并从头完成其余流程；此结果不得表述为“与历史 EGFR/VEGFR2 训练行逐行相同”。

在 PowerShell 中，与第二个入口等价的命令是：

```powershell
pwsh ./scripts/reproduce_all.ps1 -Stage All -AllowRecoveredEgfrVegfr2
```

阶段顺序为：环境 → 预检 → 四个RF预测器 → 共享增强POLYGON VAE → REINVENT4/DrugEx v2/MO-LSO底层模型 → 两靶点六方法十种子正式生成与评价 → V4-B注册消融实验 → 每种子Top10双靶点对接 → 论文Table 2和Table 3。

默认一次最多并行2个正式生成任务；可用 `-MaxParallel 3` 调整。所有阶段按固定输出路径断点续跑。只检查而不执行：

```powershell
pwsh ./scripts/reproduce_all.ps1 -Stage All -AllowRecoveredEgfrVegfr2 -DryRun
```

也可单独运行 `-Stage Predictors`、`VAE`、`BottomModels`、`Generation`、`Ablations`、`Docking` 或 `Tables`。

## 目录

| 目录 | 内容 |
|---|---|
| `method/` | CLOVER-Mol V4-B源码 |
| `baselines/*_adapter/` | POLYGON、REINVENT4、DrugEx v2、MO-LSO、GraphPareto–NSGA-II独立源码/适配器 |
| `vendor/polygon-main/` | 共享VAE的历史训练实现 |
| `data/` | 共同生成训练集与四靶点预测器数据 |
| `config/` | 全部机器可读训练/实验参数 |
| `scripts/` | 一键编排、环境、预测器、VAE、正式生成和表格入口 |
| `evaluation/`、`analysis/` | 单种子指标、质量约束、统计和主表计算 |
| `docking/`、`tools/autodock_vina_1.1.2/` | 每种子Top10对接源码、受体和第三方Vina放置说明 |
| `docs/training_configs/` | 每个方法的完整参数说明 |
| `reference_results/` | 仅保留的小型论文表和对接汇总参考值 |

不包含已退役基线。正式方法是 CLOVER-Mol 加五个基线：POLYGON（共享增强VAE）、REINVENT4、DrugEx v2、MO-LSO、GraphPareto–NSGA-II。

## 固定协议

- 共同生成数据：`data/train_smiles_only.txt`，1,584,663行，SHA-256 `4301e7f6118839465012eb93510328681ef4b7b24642e8748c4ad40971f4a304`。
- 两组靶点：EGFR/VEGFR2、PARP1/BRD4。
- 正式种子：42–51；每种子oracle预算10,240。
- 四个RF：chiral Morgan radius=2、2048 bits；RandomForestRegressor 1000树、max_features=1.0、min_samples_leaf=1、random_state=0。
- 对接：每个方法×靶点对×种子独立Top10，共1,200个候选、2,400个Vina任务。
- 统计：先逐种子计算，再报告10种子均值±样本标准差；禁止合并分子池后伪造种子统计。

完整训练参数见 [配置索引](docs/training_configs/README.md)，当前不可消除的边界见 [已知限制](docs/KNOWN_LIMITATIONS.md)，数据/代码发布边界见 [可用性说明](docs/DATA_AND_CODE_AVAILABILITY.md)。
