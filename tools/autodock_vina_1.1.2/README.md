# AutoDock Vina 1.1.2 外部运行时

论文历史对接结果使用 AutoDock Vina 1.1.2。旧版Windows二进制附带的转让条款不适合由本公共仓库再次分发，因此这里只保留放置说明。

请自行合法取得对应版本，并采用以下任一方式：

1. 设置环境变量 `VINA_EXECUTABLE` 为可执行文件绝对路径；
2. 将其放置为 `tools/autodock_vina_1.1.2/vina.exe`。

预检脚本会在正式运行前核对。若使用当前官方Apache-2.0版本代替1.1.2，必须在论文/复现记录中写明实际版本，且不能宣称对接分数与历史1.1.2逐值一致。

官方项目：https://github.com/ccsb-scripps/AutoDock-Vina
