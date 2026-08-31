# GraphPareto--NSGA-II正式搜索配置

## 方法身份

- 上游仓库：`https://github.com/Jonas-Verhellen/MolecularGraphPareto`。
- 锁定commit：`826e533b1b3995a8944e7c5cefe087806ff8c03f`。
- 环境：Python 3.10.20，`environments/requirements_snapshot_graphpareto_nsga2.txt`。
- 该方法直接进行图变异、交叉和NSGA-II选择，没有底层神经生成模型训练阶段。

## 正式参数

| 参数 | 值 |
|---|---:|
| Dataset | `data/train_smiles_only.txt` |
| Initial population | 100 |
| Survivor population | 100 |
| Offspring batch | 20 |
| Budget | 10,240终端提案 |
| Seeds | 42--51 |
| Oracle threads | 启动时显式传入，推荐4 |
| Resume | true |
| Max stalled generations | 100 |
| 目标处理 | 两个原始预测值直接Pareto最大化，不标量化 |
| Survivor selection | 非支配层优先＋crowding distance |
| Native arbiter | Glaxo、halogenicity、Veber、MW/logP/TPSA/MR |

## 必须保留的上游文件

除NSGA-II Python源码外，还必须保留：

```text
baselines/graphpareto_nsga2_adapter/upstream_official/nsga-ii/data/smarts/mutation_collection.tsv
```

不能在精简包中把整个 `nsga-ii/data/` 无条件删除，否则变异器无法启动。官方D2/HT2A模型、GuacaMol数据和NSGA-III副本不用于本论文实验，可以删除。

## 执行入口

GraphPareto无预训练命令，直接运行：

```powershell
pwsh ./run_baseline_generation.ps1 -Method graphpareto_nsga2 -TargetPair EGFR_VEGFR2 -Budget 10240 -Seed 42 -OracleThreads 4 -Evaluate
```

## 参数证据

- `baselines/graphpareto_nsga2_adapter/adapter_optimize_dual_oracle.py`
- `results/baselines/graphpareto_nsga2/formal_10240_seed42/metadata.json`
- `config/baseline_experiments.json`

