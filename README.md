# 自身方法改造

这是从原 `D:\code` 工作区抽出的精简、可复现实验目录，只保留 EGFR/VEGFR2 双目标自身方法及其正式评估所需资产。原工作区未被修改。

## 当前方法

- 锁定 POLYGON VAE 作为分子生成底座；
- 双目标潜空间 PPO（EGFR、VEGFR2）；
- 目标独立 Multi-Critic；
- 三阶段动态目标权重；
- 多温度、多步长探索通道及自适应分配；
- K=1/3/5 真实时序潜空间轨迹，逐步状态转移并用 GAE 回传终点奖励；
- QED、SA、Novelty 作为质量审计指标，不混入双目标主优化定义。

每条轨迹只在终点解码并调用一次 EGFR/VEGFR2 oracle，因此 K=1、3、5 使用完全相同的分子评价预算。中间步骤的目标奖励为 0，终点分数由目标独立 Critic 的 GAE 向前传播。默认使用 `1/sqrt(K)` 步长归一化，降低轨迹长度与原始搜索半径混杂的问题。

这里区分两个容易混淆的概念：探索通道中的“多步长”是不同的单步尺度 multiplier；`--trajectory-length` 才是真正连续状态转移的 K 步轨迹。

## 目录

- `method/`：主运行链和论文必要消融；
- `models/`：唯一锁定 VAE 与两个 RF oracle；
- `data/`：Novelty 审计使用的规范化训练集缓存；
- `vendor/polygon-main/`：VAE 与 SA scorer 所需的最小 POLYGON 源码；
- `evaluation/`：统一 Pareto、质量、可靠性与统计评估；
- `config/`：正式预算、种子和消融矩阵；
- `docs/`：SCI Q2 基线实验计划；
- `logs/`：新实验输出，默认不纳入版本控制。

## 运行

在 PowerShell 中：

```powershell
Set-Location C:\Users\Lenovo\Documents\自身方法改造
.\run_method.ps1 -OracleBudget 2048 -Seed 42 -TrajectoryLength 3 -Device cuda
```

快速检查参数与依赖：

```powershell
D:\code\.conda-envs\drug-pareto-rl\python.exe .\method\ablation\run_wc_two_targets_multiexplore.py --help
```

正式实验按 `config/formal_experiments.json` 使用 10,240 oracle 预算和预注册种子。任何新改造先进行 2,048 预算、3 种子筛选，再进入正式实验。

轨迹消融固定其他设置，仅改变：

```powershell
.\run_method.ps1 -OracleBudget 2048 -Seed 42 -TrajectoryLength 1 -Device cuda
.\run_method.ps1 -OracleBudget 2048 -Seed 42 -TrajectoryLength 3 -Device cuda
.\run_method.ps1 -OracleBudget 2048 -Seed 42 -TrajectoryLength 5 -Device cuda
```

输出中的 `trajectory_steps`、`path_length`、`net_displacement` 和 `policy_transitions` 用于审计真实状态转移；`generated_rows` 与 `oracle_budget` 应始终相等。

## 未迁移内容

旧 VAE/历史检查点、PCGrad/CAGrad 探索分支、早期三目标 runner、近似基线实现、训练日志、图片、缓存、压缩包、论文构建脚本和重复可视化脚本均未迁移。官方基线应独立适配，避免与自身方法代码混在一起。
