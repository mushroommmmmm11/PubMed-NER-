# PubMed-NER-

PubMed 文献驱动的 NER 模型持续学习系统。

## 新增：R 网络分析脚本的 Python 版本

仓库中新增了 `analysis_network.py`，用于把你提供的主分析 R 脚本迁移为更易维护的 Python 实现，并尽量减少重复代码。

### 脚本特点

- 读取双表头 Excel，并自动清洗重复列名。
- 统一封装“读数、建网、导表、画图、bootstrap、组间比较、DAG”等重复流程。
- 使用 `GraphicalLasso` 估计偏相关网络，避免大量重复手工代码。
- 默认 bootstrap / permutation 次数比原 R 脚本更保守，运行更流畅；需要更高精度时可通过参数调大。
- 对可选依赖（如 `pgmpy`）做了显式检查，缺失时会给出清晰提示。

### 推荐依赖

```bash
pip install pandas numpy scipy scikit-learn matplotlib networkx openpyxl seaborn joblib pgmpy
```

### 用法示例

```bash
python analysis_network.py \
  --input-file /path/to/DATA.xlsx \
  --output-dir main_analysis_all_figures_py
```

### 常用可调参数

- `--bootstrap-iterations`: 非参数 bootstrap 次数。
- `--case-bootstrap-iterations`: case-dropping 稳定性分析次数。
- `--nct-iterations`: 组间网络比较置换次数。
- `--dag-bootstrap-iterations`: DAG bootstrap 次数。
- `--alpha`: Graphical Lasso 收缩强度。
- `--max-workers`: 并行 worker 数。

### 输出内容

脚本会输出以下类型的结果：

- 主网络图与偏相关矩阵。
- 中心性表、桥接中心性表与对应图形。
- bootstrap 边权区间、中心性差异结果。
- case-dropping 稳定性图和 CS 系数表。
- 二分类组间网络比较图与置换检验摘要。
- 探索性 DAG 图、弧强度表和会话信息文件。

> 注意：由于 Python 生态与 R 的 `bootnet` / `qgraph` / `NetworkComparisonTest` 并不完全等价，当前实现优先保证流程清晰、可维护、运行稳定，再尽量对齐原分析思路。
