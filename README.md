# pandas 决策树 → GBDT（UCI Breast Cancer / Ljubljana）

本仓库最初是一个用 pandas 手写 ID3 决策树的教学实验。本次改造做了三件事：

1. **修复原实现中会导致数据泄漏、空分支崩溃、随机预测、剪枝错误和低效计算的 bug**；
2. **从零实现一个正则化的 GBDT（二阶牛顿提升 + logistic loss）**，不依赖 scikit-learn；
3. **给出可复现的诚实评估**：分层 holdout + 嵌套交叉验证 + scikit-learn 天花板对照。

> ⚠️ **关于“90% 正确率”的重要说明**
>
> 本仓库自带的 `breast-cancer.data` 是 UCI **Ljubljana 复发数据集**：286 条样本、9 个类别特征，类别为
> `no-recurrence-events`(201) / `recurrence-events`(85)。该数据集噪声大、类别高度重叠。
> **在无泄漏评估下，现有方法的准确率集中在 72%–78%**（数据集说明中历史工作为 66%–78%）。
> **90% 的测试准确率在这个数据集上没有已知的无泄漏方法可以达到**；这不是调参或换基模型能解决的。
>
> 本项目通过实验证明了这一点：
> - 本实现 GBDT 的嵌套 5 折 ×3 交叉验证准确率：**0.7341 ± 0.0346**；
> - scikit-learn RandomForest 的重复 5 折 CV：**0.7484 ± 0.0490**；
> - scikit-learn GradientBoosting：**0.7152**；
> - 57 条 holdout 的 95% bootstrap 置信区间：**[0.561, 0.807]**，上界也远低于 0.90。
>
> 把 GBDT 的树加到 300 棵，**训练集准确率可以达到 92.1%**（满足字面上的“90%”），
> 但同一个模型的**测试准确率只有 64.9%**——这是记忆，不是泛化。
> 本项目不会用测试集调参、不会泄漏标签、也不会用训练集准确率冒充测试准确率。
>
> 如果你的 90% 要求针对的是**更容易的 Wisconsin 乳腺癌数据集**（569 条、30 个连续特征），
> 运行 `python wisconsin_demo.py`：同一个 GBDT 实现可达到 **97.35% 测试准确率 / 95.95% 五折 CV**。

---

## 0. 数据审计：一个容易制造“90% 假象”的陷阱

原始 `breast-cancer.data` **按类别排序**：前 201 行全部是 `no-recurrence-events`，
后 85 行全部是 `recurrence-events`（只有一个标签变化点，位置 201）。

这意味着：

- 如果**不 shuffle** 就按前 80% / 后 20% 划分，训练集和测试集类别不重叠，准确率可以轻易达到 100%；
- 如果把**行号**当特征，树只需在 201 附近切一刀就能在测试集上接近 100%；
- 如果 K 折划分不 shuffle，某些折可能只含一个类别。

这些都不是模型能力，而是**数据集排序泄漏**。本项目**显式禁止**这类做法：

- `stratified_split` / `stratified_kfold` 先按类别分组、类内随机置换，再分配索引；
- 行号/索引从不作为特征；
- 测试集只在最终评估时使用。

如果你看到的 90%+ 结果来自“不 shuffle”或“行号特征”，请把它视为泄漏而不是性能。

---

## 1. 快速开始

```bash
# 只依赖 numpy + pandas
python -m pip install -r requirements.txt

# 完整实验：修复后的 ID3/C4.5 + GBDT + 嵌套 CV
python ml_exp.py

# 额外运行 scikit-learn 天花板对照（需要 scikit-learn）
python -m pip install -r requirements-optional.txt
python ml_exp.py --benchmark

# 在更容易的 Wisconsin 数据集上演示 >90%（需要 scikit-learn）
python wisconsin_demo.py

# 单元测试
python -m unittest discover -s tests -v
```

输出文件写入 `results/`：

| 文件 | 说明 |
|---|---|
| `metrics.json` | 全部指标、参数、数据划分、目标检查 |
| `id3_rules.txt` | 修复后 ID3/C4.5 的规则（与原格式一致） |
| `id3_tree.txt` | 可读的树结构 |
| `id3_model.pkl` / `gbdt_model.pkl` | 拟合后的模型（含 metadata） |
| `gbdt_feature_importance.csv` | GBDT 特征重要度 |
| `wisconsin_metrics.json` / `wisconsin_gbdt_model.pkl` | Wisconsin 演示 |

---

## 2. 原实现的 bug 与修复

| # | 原实现的问题 | 后果 | 修复 |
|---|---|---|---|
| 1 | `divide_train_test` 用**闭区间** `.loc[0:threshold]` 切片，且 `train2` 从 `threshold_train` 开始 | train1 / train2 重叠；`fraction=1` 时 `test` 为空。原脚本末尾实际打印的是 `train2` 的准确率，**从未在测试集上评估** | `stratified_split()` 用互斥索引，保证三份不重叠、每类分层、每份非空；测试集真正留出 |
| 2 | `if examples.empty:` 检查了**父节点**，而应该检查 `examples_vi.empty`；而且对空集调用 `(target == positive).all()` 会因 vacuous truth 返回 `True`，`mode()[0]` 分支则可能崩溃 | 空分支会被误判为 positive 叶，或触发 `mode()` 异常 | 在递归入口拒绝空集；分支为空时使用父节点保存的多数类作为确定性回退 |
| 3 | 构建时把 `?` 分支从 `test_branch` 中删除；预测时对缺失值 `random.choice` | 缺失样本训练/预测路径不一致；**同一模型两次预测可能不同** | `?` 作为显式类别参与分裂；未见过的类别回退到节点保存的多数类，完全确定 |
| 4 | 剪枝在每个节点上重新计算**整棵树**在验证集上的准确率，且 `id3_correctness` 用 `.loc[i]` 逐行访问 | 复杂度约 `O(节点数 × 样本数)`；非 `RangeIndex` 时行为错误；验证集同时用于剪枝和评估，指标乐观 | 自底向上 reduced-error pruning：只把到达该节点的验证索引传下去，一次遍历完成，返回剪枝后的预测 |
| 5 | `information_gain` 对每个属性、每个取值反复 `s[s[attr] == v]` 过滤 DataFrame | `O(属性数 × 取值数 × 样本数)`，且每次创建新 DataFrame | 用 `np.bincount` 构造列联表，一次算出加权熵/增益/增益率 |
| 6 | `entropy` 用 `unique()` + Python 循环 | 慢且重复扫描 | 从计数向量直接计算，向量化 |
| 7 | 分类逐行 `example_test.loc[i, :]`，每行一次 Python 分类 | 预测慢；依赖索引连续 | 批量数组遍历，`predict_proba` 一次性返回 |
| 8 | 没有 `max_depth` / `min_samples_leaf` / 最小增益等预剪枝；剪枝只在验证集上贪婪比较 | 容易过拟合；小样本上不稳定 | 增加深度、叶大小、最小增益控制；剪枝自底向上且局部化 |
| 9 | 全局 `attributes` 被 `remove()` 原地修改；`sample(frac=1)` 无随机种子 | 不可复现；重复调用状态污染 | 显式传入特征元数据；所有随机性使用 `default_rng(seed)` |
| 10 | `print_tree` 用 `'\b\b\b\b'` 控制终端显示 | 输出到文件/管道时出现退格字符 | 结构化树输出 + 规则文件导出，无控制字符 |
| 11 | `save_tree` 直接 pickle 树对象，无格式/元数据；`load_tree` 无校验 | 版本不兼容、难以追溯 | `save_model/load_model` 带格式版本与 metadata |
| 12 | 无最终测试集、无交叉验证、无基线 | 无法判断泛化能力 | 分层 holdout + 嵌套 CV + 多数类基线 + sklearn 天花板对照 |

### 性能改进（实测，286 行）

| 操作 | 原实现 | 新实现 |
|---|---|---|
| ID3 建树（同一台机器） | 原实现端到端约 0.60 s（含启动/IO，且依赖随机路径） | **约 0.004 s**（仅 `fit`，不含启动/IO） |
| ID3 预测 | 逐行 `.loc`，每次重新遍历 | 批量数组预测，~毫秒级 |
| GBDT 训练（64 棵树，早停） | 无 | **约 0.13 s** |
| 嵌套 5×3 CV | 无 | **约 3 s** |

注：0.60 s 是原脚本端到端耗时（含解释器启动、CSV 读取、打印和保存），0.004 s 是单次 ID3 `fit`，两者不是严格同口径。
但算法层面的差异是明确的：原实现每次增益都重复过滤 DataFrame，复杂度 `O(d·v·n)`；
新实现用一次列联表统计，复杂度 `O(d·n)`（`v` 为取值数），并且 GBDT 的节点分裂用排序 + 前缀和完成。

---

## 3. 项目结构

```
pandas_dt/
├── data.py          # 数据加载、缺失值、类别编码、分层切分、K 折
├── tree.py          # 修复后的 ID3/C4.5（多路类别分裂 + REP 剪枝）
├── gbdt.py          # CART 回归树 + 二阶 GBDT（logistic loss）
├── metrics.py       # accuracy / logloss / AUC / bootstrap CI / 阈值调优
└── utils.py         # 模型序列化、JSON 报告
ml_exp.py            # 主实验入口（ID3 + GBDT + 嵌套 CV + 目标检查）
wisconsin_demo.py    # 可选：同一个 GBDT 在 Wisconsin 数据集上的 >90% 演示
tests/               # 32 个单元测试
```

### ID3 / C4.5

- 多路类别分裂；`criterion="gain"`（ID3）或 `"gain_ratio"`（C4.5，默认，降低高基数属性偏置）。
- `?` 是显式分支；未见类别用节点多数类回退，预测确定。
- 预剪枝 + 验证集 reduced-error pruning。
- 提供 `decision_rules()` / `export_rules()` / `str(tree)`。

### GBDT

二分类 logistic loss，二阶牛顿提升，目标函数：

```
Obj = Σ [ g_i f(x_i) + 1/2 h_i f(x_i)^2 ] + λ/2 Σ w_j^2
g_i = p_i - y_i,  h_i = p_i(1-p_i)
split gain = G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)
leaf value = -G/(H+λ)
```

支持：`n_estimators`、`learning_rate`、`max_depth`、`min_samples_leaf`、`min_child_weight`、
`reg_lambda`、`subsample`、`max_features`、`early_stopping_rounds`。
行采样、列采样、早停均使用固定随机种子。

特征编码（`encode_numeric`）：
- 有序属性（age / tumor-size / inv-nodes / deg-malig）保留**序数阈值**；
- 无序属性（menopause / node-caps / breast / breast-quad / irradiat）one-hot，`?` 单独一列；
- 共 21 维，无 NaN、无隐式填充。

---

## 4. 本数据集上的结果（可复现）

命令：`python ml_exp.py --benchmark`

划分：分层 60/20/20，`seed=42`，train=172 / val=57 / test=57。

| 模型 | 训练准确率 | holdout 测试准确率 | 备注 |
|---|---:|---:|---|
| 多数类基线 | — | 0.7018 | 全部预测 no-recurrence |
| ID3/C4.5（修复后，REP 剪枝） | 0.7500 | 0.6842 | val 0.7368；25 节点/18 叶 |
| GBDT（早停，train-only） | 0.8081 | 0.7018 | 64/94 棵树 |
| GBDT（train+val 重训） | 0.7948 | 0.6842 | 推荐保存的模型 |
| GBDT 嵌套 5 折 ×3 CV | — | **0.7341 ± 0.0346** | 无泄漏（内层选树数/阈值） |
| GBDT 高容量（300 棵，无早停） | **0.9214** | 0.6491 | **训练 >90%，测试反而下降** |
| sklearn RandomForest（500 棵，one-hot） | — | 0.7484 ± 0.0490 | 重复 5 折 CV |
| sklearn GradientBoosting | — | 0.7152 ± 0.0695 | 重复 5 折 CV |
| sklearn LogisticRegression | — | 0.6888 ± 0.0452 | 重复 5 折 CV |

> 由于只有 286 条样本，单次 holdout 的方差很大；**嵌套 CV 是更可信的估计**。
> 57 条测试集的 95% bootstrap CI 为 `[0.561, 0.807]`，说明 0.90 不在合理范围内。

### 为什么 90% 测试准确率不可能？

1. **数据量**：286 条、9 个离散特征，其中 `node-caps` 8 个缺失、`breast-quad` 1 个缺失。
2. **类别重叠**：大量相同的特征组合对应不同标签；数据说明中历史方法的准确率范围是 66%–78%
   （Assistant-86 为 78%）。
3. **统计检验**：本实现嵌套 CV 为 73.4%，最强 sklearn 基线为 74.8%；要在一份 57 样本的测试集上
   达到 90%，需要至少 52 条正确，比当前最好结果多约 10 条。置信区间上界 0.807 也达不到。
4. **训练集 90% 是记忆**：300 棵树的训练准确率为 92.1%，但测试降到 64.9%。把树继续加深/加多只会
   让训练准确率更高、测试更差。

因此，本项目**没有**通过以下方式伪造 90%：
- 用测试集调参；
- 在划分时让 train/test 重叠（原实现的问题）；
- 把训练准确率当作测试准确率；
- 删除困难样本或只报告高置信度子集。

如果业务上必须达到 90% 的**测试**准确率，可行方向是：
- 使用信号更强的数据集（如 Wisconsin Diagnostic，本项目演示 97.35%）；
- 增加样本量 / 更有判别力的特征；
- 如果允许拒答（rejection option），可以报告“高置信度子集上的准确率”，但必须同时报告覆盖率。

### 同一个 GBDT 在 Wisconsin 数据集上

```
python wisconsin_demo.py
```

- 569 条、30 个连续特征、二分类；
- holdout 测试准确率：**0.9735**，95% CI `[0.938, 1.000]`；
- 固定树数的 5 折 CV：**0.9595 ± 0.0155**；
- 训练准确率：1.0000；
- 说明 GBDT 实现本身是正确的，**限制来自 Ljubljana 数据集本身**。

> Wisconsin 演示仅用于证明实现能力，**不能**当作 `breast-cancer.data` 的结果。

---

## 5. 评估协议

- **Holdout**：`stratified_split(y, test_size=0.2, val_size=0.2, seed=42)`，互斥、分层。
- **阈值**：只在验证集上选择，再应用到测试集；`accuracy@0.5` 是主指标。
- **嵌套 CV**：
  1. 外层 5 折；
  2. 外层训练集再分层切 80/20，内层验证集用于早停和阈值选择；
  3. 用选出的树数在外层训练集上重训；
  4. 在外层测试折上评估，重复 3 次。
- **高容量对照**：单独训练 300 棵树、不早停，仅用于展示训练/测试的背离。
- **sklearn 天花板**：`RepeatedStratifiedKFold(5 折, 2 次)`，one-hot 全类别编码，包含多数类、
  逻辑回归、随机森林、梯度提升。

---

## 6. 复现性与测试

```bash
python -m unittest discover -s tests -v
# Ran 32 tests ... OK
```

测试覆盖：
- 数据加载/缺失值/三种编码；
- 切分互斥性、分层、K 折划分（原数据泄漏 bug 的回归测试）；
- ID3 完美分裂、单类、缺失/未见类别确定性、剪枝、规则导出、非法输入；
- GBDT sigmoid 数值稳定、XOR 学习、早停、单类、特征重要度、序列化；
- 指标（accuracy/confusion/AUC/bootstrap/阈值）；
- 真实数据 smoke test。

---

## 7. 参考文献与数据来源

- UCI Machine Learning Repository: Breast Cancer dataset
  <https://archive.ics.uci.edu/dataset/14/breast+cancer>
- 本地 `breast-cancer.names` 中的历史结果：Michalski et al. 66%–72%；Clark & Niblett 65%–72%；
  Tan & Eshelman 68%–73.5%；Assistant-86 78%。
- scikit-learn Wisconsin Diagnostic Breast Cancer:
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html>

---

## 8. 附注：旧的 `rules0.7913043478260869.txt`

该文件来自原实现，准确率是在 `train2`（剪枝集）上计算的；原实现的 train1/train2/test 存在重叠，
且 `test` 为空。因此这个 0.79 **不是测试准确率**，不能与本文的 holdout/CV 数字直接比较。
新实现生成的是 `results/id3_rules.txt` 和 `results/metrics.json`。
