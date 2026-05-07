# envmolbench

**面向分子性质预测的基准测试框架，内置 45 个环境化学数据集。**

[![PyPI version](https://img.shields.io/badge/pypi-0.1.2-blue)](https://pypi.org/project/envmolbench/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**[English Documentation](README.md)**

`envmolbench` 提供统一的 API，用于加载分子数据集、划分数据、训练模型、评估预测结果，以及计算结构-活性景观粗糙度（roughness）和共形不确定性校准（conformal prediction）。只需 `import envmolbench as eb` 即可使用全部功能。

**项目网站：** [www.ai4env.world/envmolbench](https://www.ai4env.world/envmolbench/)  
**GitHub：** [github.com/AI4-LH/envmolbench](https://github.com/AI4-LH/envmolbench)

---

## 功能特性

- **45 个内置数据集**，涵盖毒性、ADME 性质、环境化学、分子物化性质等
- **14 种模型**：随机森林、XGBoost、CatBoost、SVM、Ridge、Lasso、逻辑回归、GNN、GCN、ChemBERTa、Chemprop、UniMol、CNN
- **7 种特征化方法**：Morgan 指纹、MACCS 键、Morgan Count、Mordred 描述符、图表示、分子图像、SMILES
- **5 种数据拆分策略**：Scaffold 拆分、Butina 聚类、MaxMin 采样、随机拆分、时间序列拆分
- **景观粗糙度分析**：分类任务（NNDR）和回归任务（SALI），支持 4 种指纹类型
- **共形预测与 ECE**：回归和分类任务的不确定性校准
- **超参数优化**（基于 Optuna）
- **CLI 命令行**支持单数据集实验和批量实验

---

## 安装

```bash
# 核心安装（RF、XGBoost、SVM、Ridge、Lasso、逻辑回归）
pip install envmolbench

# 核心 + CatBoost
pip install envmolbench[catboost]

# 核心 + 超参数优化（Optuna）
pip install envmolbench[optuna]

# 核心 + 深度学习模型（ChemBERTa、Chemprop、GNN、CNN、UniMol）
# 注意：torch-geometric 需与安装的 PyTorch 版本匹配
# 参考：https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install envmolbench[dl]

# 核心 + Mordred 描述符特征化方法
pip install envmolbench[mordred]

# 完整安装（所有依赖）
pip install envmolbench[all]
```

---

## 数据集配置

内置数据集**未打包**在 pip 包中（体积过大）。需要单独提供数据集文件。

**方式一 — 设置环境变量（pip 安装后推荐）：**
```bash
# Linux / macOS
export ENVMOLBENCH_DATA_DIR=/path/to/your/datasets

# Windows PowerShell
$env:ENVMOLBENCH_DATA_DIR = "C:\path\to\your\datasets"
```

**方式二 — 直接传入 `datasets_dir` 参数：**
```python
smiles, labels, task = eb.load_dataset("esol", datasets_dir="/path/to/datasets")
```

**方式三 — 开发模式（克隆仓库）：**  
直接克隆仓库后，加载器会自动找到与包同级的 `datasets/` 目录。

> 数据集文件可从项目网站或 GitHub Releases 下载。

**方式四 — 通过 API 下载（推荐）：**
```python
import envmolbench as eb

# 下载单个数据集（默认保存到 ~/.envmolbench/datasets/）
eb.download_dataset("hlm")
eb.download_dataset("hlm", save_dir="/my/data")  # 指定目录

# 一次性下载全部 45 个数据集
eb.download_all_datasets()
eb.download_all_datasets(save_dir="/my/data", overwrite=False)
```

下载完成后，将数据目录指向加载器：
```bash
# Linux / macOS
export ENVMOLBENCH_DATA_DIR=~/.envmolbench/datasets

# Windows PowerShell
$env:ENVMOLBENCH_DATA_DIR = "$env:USERPROFILE\.envmolbench\datasets"
```

---

## 快速开始

### 加载内置数据集

```python
import envmolbench as eb

smiles, labels, task = eb.load_dataset("esol")
print(task)       # "regression"
print(len(smiles))  # 分子数量
```

### 加载本地 CSV 文件

```python
# 自动识别 SMILES 列（列名含 "smi"）和标签列
smiles, labels, task = eb.load_dataset("my_data.csv")

# 手动指定列名
smiles, labels, task = eb.load_dataset(
    "my_data.csv",
    smiles_col="molecule",
    label_col="toxicity",
    task_type="classification",
)
```

### 一键训练

```python
results = eb.quick_train(model="rf", dataset="esol", split="scaffold")
print(results)
# {"train_rmse": 0.41, "val_rmse": 0.67, "test_rmse": 0.72, ...}
```

### 分步流程

```python
import envmolbench as eb

smiles, labels, task = eb.load_dataset("hepatotoxicity")

# 数据拆分
splits = eb.split_data(smiles, labels, method="scaffold")

# 获取模型
model = eb.get_model("rf", task_type=task)

# 训练与预测
model.fit(splits.train_smiles, splits.train_labels,
          splits.val_smiles,   splits.val_labels)
preds = model.predict(splits.test_smiles)

# 评估指标
metrics = eb.calc_metrics(splits.test_labels, preds, task)
print(metrics)
# {"auc": 0.83, "accuracy": 0.76, "f1": 0.74, ...}
```

### 交叉验证

```python
import envmolbench as eb

smiles, labels, task = eb.load_dataset("hepatotoxicity")

# 5 折 Scaffold 交叉验证
cv_splits = eb.split_data(smiles, labels, method="scaffold", n_folds=5)
for fold in cv_splits:
    model = eb.get_model("rf", task_type=task)
    model.fit(fold.train_smiles, fold.train_labels,
              fold.val_smiles,   fold.val_labels)
    preds = model.predict(fold.test_smiles)
    print(eb.calc_metrics(fold.test_labels, preds, task))
```

### 景观粗糙度分析

```python
import envmolbench as eb

smiles, labels, task = eb.load_dataset("hepatotoxicity")

# 单种指纹（默认：ecfp4）
r = eb.roughness((smiles, labels, task))
print(r)
# {"task_type": "classification", "fp_type": "ecfp4",
#  "metric": "NNDR", "mean": 0.312, "std": 0.015, "n_valid": 450}

# 同时计算四种指纹
r = eb.roughness(
    (smiles, labels, task),
    fp_type=["ecfp4", "maccs", "rdkit_topo", "atompair"],
)
print(r)
# {"ecfp4": {...}, "maccs": {...}, "rdkit_topo": {...}, "atompair": {...}}

# 直接传入 smiles 和 labels
r = eb.compute_roughness(smiles, labels, task_type="classification")

# 底层函数
nndr_mean, nndr_std, n = eb.compute_nndr(smiles, labels, fp_type="ecfp4",
                                          max_samples=1000, n_repeats=5)
sali_mean, sali_std,  n = eb.compute_sali(smiles, labels, fp_type="ecfp4")
```

**粗糙度参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `fp_type` | `"ecfp4"` | 指纹类型：字符串或列表 |
| `max_samples` | `2000` | 子采样前的最大分子数（NNDR） |
| `n_repeats` | `10` | 子采样重复次数（用于方差估计） |
| `seed` | `42` | 随机种子 |

### 共形预测与不确定性校准

```python
import numpy as np
import envmolbench as eb

# --- 分类任务 ---
y_true = np.array([1, 0, 1, 0, 1, 0] * 20)
y_prob = np.array([0.85, 0.15, 0.78, 0.22, 0.91, 0.08] * 20)

result = eb.conformal_prediction(y_true, y_prob, task_type="classification",
                                  nominal_level=0.90, n_repeats=10)
print(result)
# {"task_type": "classification", "nominal_level": 0.9,
#  "coverage_mean": 0.912, "coverage_std": 0.018,
#  "calibration_factor": None, "ece": 0.041, "n_samples": 120}

ece = eb.compute_ece(y_true, y_prob, task_type="classification")

# --- 回归任务 ---
y_true_reg = np.random.randn(200)
y_pred_reg = y_true_reg + np.random.randn(200) * 0.5
y_std_reg  = np.abs(np.random.randn(200)) * 0.3

result = eb.conformal_prediction(y_true_reg, y_pred_reg, y_std=y_std_reg,
                                  task_type="regression", nominal_level=0.90)
print(result)
# {"task_type": "regression", "nominal_level": 0.9,
#  "coverage_mean": 0.903, "coverage_std": 0.021,
#  "calibration_factor": 1.02, "ece": 0.038, "n_samples": 200}

ece = eb.compute_ece(y_true_reg, y_pred_reg, y_std=y_std_reg, task_type="regression")
```

---

## 内置数据集（45 个）

### 分类数据集 — 毒性与生物活性（17 个）

| 数据集名称 | 描述 |
|---|---|
| `hepatotoxicity` | 肝毒性 |
| `Hepatotoxicants` | 肝毒物分类 |
| `androgen_receptor` | 雄激素受体结合活性 |
| `estrogen_receptor` | 雌激素受体 α 结合 |
| `antibacterial_activity` | 抗菌活性 |
| `ames_mutagenicity` | Ames 致突变性 |
| `carcinogenicity` | 致癌性 |
| `oral_carcinogenicity` | 口服致癌性 |
| `cytotoxicity` | 细胞毒性 |
| `neurotoxicity` | 神经毒性 |
| `ocular_toxicity` | 眼部（眼睛）毒性 |
| `prenatal_development` | 产前发育毒性 |
| `reproductive_toxicity` | 生殖毒性 |
| `respiratory_toxicity` | 呼吸系统毒性 |
| `skin_corrosion` | 皮肤腐蚀性 |
| `pbt` | 持久性、生物累积性和毒性（PBT） |
| `tshr_agonist` | TSHR 激动剂活性 |

### 回归数据集 — 物理化学与 ADME 性质（14 个）

| 数据集名称 | 描述 |
|---|---|
| `esol` | 水溶性（log mol/L） |
| `freesolv` | 水化自由能（kcal/mol） |
| `lipophilicity` | 亲脂性（log D） |
| `solubility` | 热力学溶解度 |
| `fubrain` | 脑组织中未结合分数 |
| `fup` | 血浆中未结合分数 |
| `clin_fup` | 临床血浆未结合分数 |
| `hlm` | 人肝微粒体清除率 |
| `oral_bioavailability` | 口服生物利用度 |
| `pka_acidic` | 酸性 pKa |
| `kh` | 亨利定律常数 |
| `pparg_ic50` | PPARγ IC50 |
| `pparg_pkd` | PPARγ pKd |
| `pfas` | 全氟及多氟烷基物质（PFAS）环境性质 |

### 环境化学数据集（9 个）

| 数据集名称 | 描述 |
|---|---|
| `aero_bio_c` | 好氧生物降解（分类） |
| `aero_bio_r` | 好氧生物降解（回归） |
| `tfishbio` | 鱼类生物浓缩因子 |
| `koa` | 辛醇-空气分配系数 |
| `koc` | 有机碳分配系数 |
| `koc2` | 有机碳分配系数（替代数据集） |
| `tbp` | 磷酸三丁酯分配 |
| `si_oh` | OH 自由基速率常数 |
| `si_so4` | SO₄ 自由基速率常数 |

### 其他分子性质（5 个）

| 数据集名称 | 描述 |
|---|---|
| `ccs_mh` | 碰撞截面 [M+H]⁺ |
| `ccs_mna` | 碰撞截面 [M+Na]⁺ |
| `fba` | 分子结合亲和力 |
| `fbc` | 分子结合常数 |
| `plv` | 肺部分配体积 |

---

## 可用模型（14 种）

| 模型名称 | 类型 | 安装要求 |
|---|---|---|
| `rf` / `random_forest` | 随机森林 | 核心安装 |
| `xgboost` | XGBoost | 核心安装 |
| `catboost` | CatBoost | 核心安装 |
| `svc` / `svr` | 支持向量机 | 核心安装 |
| `ridge` | Ridge 回归 | 核心安装 |
| `lasso` | Lasso 回归 | 核心安装 |
| `logistic_regression` | 逻辑回归 | 核心安装 |
| `gnn` | 图神经网络（GNN） | `pip install envmolbench[dl]` |
| `gcn` | 图卷积网络（GCN） | `pip install envmolbench[dl]` |
| `chemberta` | Transformer（SMILES 序列） | `pip install envmolbench[dl]` |
| `chemprop` | 消息传递神经网络（MPNN） | `pip install envmolbench[dl]` |
| `unimol` | 3D 构象模型 | `pip install envmolbench[dl]` |
| `cnn` | 基于 ResNet18 的分子图像模型 | `pip install envmolbench[dl]` |

```python
print(eb.list_models())
```

---

## 可用特征化方法（7 种）

| 特征化名称 | 描述 | 安装要求 |
|---|---|---|
| `morgan` | Morgan（ECFP4）二进制指纹 | 核心 |
| `morgan_count` | Morgan 计数指纹 | 核心 |
| `maccs` | MACCS 167 位键 | 核心 |
| `mordred` | ~1800 种 Mordred 描述符 | `pip install envmolbench[mordred]` |
| `graph` | GNN 图表示 | `pip install envmolbench[dl]` |
| `image` | CNN 分子图像 | `pip install envmolbench[dl]` |
| `smiles` | 原始 SMILES 序列 | `pip install envmolbench[dl]` |

```python
featurizer = eb.get_featurizer("morgan")
X = featurizer.fit_transform(smiles)

print(eb.list_featurizers())
```

---

## 数据拆分方法（5 种）

| 方法 | 参数值 | 描述 |
|---|---|---|
| Scaffold 拆分 | `scaffold` | 基于 Murcko 骨架（默认，推荐） |
| 随机拆分 | `random` | 标准分层随机拆分 |
| Butina 聚类 | `butina` | 基于相似性阈值的聚类拆分 |
| MaxMin 采样 | `maxmin` | 最大化多样性的差异性拆分 |
| 时间序列 | `time` | 按时间顺序拆分（需要时间列） |

```python
splits = eb.split_data(smiles, labels, method="scaffold")
# splits.train_smiles, splits.val_smiles, splits.test_smiles
# splits.train_labels, splits.val_labels, splits.test_labels
```

---

## 命令行用法

安装后提供两个命令行工具：

```bash
# 单数据集实验
envmolbench --model rf --data esol --split scaffold

# 批量实验（遍历所有数据集）
envmolbench-batch --model rf,xgboost --split scaffold
```

---

## API 参考

| 函数 | 描述 |
|---|---|
| `eb.download_dataset(name, ...)` | 从官网下载单个内置数据集 |
| `eb.download_all_datasets(...)` | 从官网下载全部 45 个数据集 |
| `eb.load_dataset(name_or_path, ...)` | 加载内置或本地 CSV 数据集 |
| `eb.list_datasets()` | 列出所有内置数据集名称 |
| `eb.split_data(smiles, labels, method, n_folds)` | 拆分数据（单次或 k 折交叉验证） |
| `eb.get_model(name, task_type)` | 按名称获取模型 |
| `eb.list_models()` | 列出所有可用模型名称 |
| `eb.get_featurizer(name)` | 按名称获取特征化方法 |
| `eb.list_featurizers()` | 列出所有特征化方法 |
| `eb.calc_metrics(y_true, y_pred, task)` | 计算评估指标 |
| `eb.quick_train(model, dataset, split)` | 一键训练流程 |
| `eb.roughness(dataset, fp_type, ...)` | 计算景观粗糙度（NNDR/SALI） |
| `eb.compute_roughness(smiles, labels, ...)` | 从独立数组计算粗糙度 |
| `eb.compute_nndr(smiles, labels, fp_type)` | NNDR（分类粗糙度） |
| `eb.compute_sali(smiles, labels, fp_type)` | SALI（回归粗糙度） |
| `eb.conformal_prediction(y_true, y_pred, ...)` | 共形预测（回归/分类） |
| `eb.compute_ece(y_true, y_pred, ...)` | 期望校准误差（ECE） |

---

## 引用

如果您在研究中使用了 `envmolbench`，请引用：

```bibtex
@software{envmolbench2025,
  title   = {envmolbench: A Molecular Property Prediction Benchmarking Framework},
  author  = {EnvMolBench Team},
  year    = {2025},
  url     = {https://github.com/AI4-LH/envmolbench},
  version = {0.1.0},
}
```

---

## 许可证

MIT 许可证。详见 [LICENSE](LICENSE) 文件。
