# envmolbench

**A molecular property prediction benchmarking framework with 45 built-in environmental chemistry datasets.**

[![PyPI version](https://img.shields.io/badge/pypi-0.1.2-blue)](https://pypi.org/project/envmolbench/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**[中文文档](README_zh.md)**

`envmolbench` provides a unified API for loading molecular datasets, splitting data, training models, evaluating predictions, and computing landscape roughness and conformal uncertainty calibration — all with a single `import envmolbench as eb`.

**Website:** [www.ai4env.world/envmolbench](https://www.ai4env.world/envmolbench/)  
**GitHub:** [github.com/AI4-LH/envmolbench](https://github.com/AI4-LH/envmolbench)

---

## Features

- **45 built-in datasets** covering toxicity, ADME, environmental chemistry, and molecular properties
- **14 models**: Random Forest, XGBoost, CatBoost, SVM, Ridge, Lasso, Logistic Regression, GNN, GCN, ChemBERTa, Chemprop, UniMol, CNN
- **7 featurizers**: Morgan, MACCS, Morgan Count, Mordred descriptors, Graph, Molecular Image, SMILES
- **5 splitting strategies**: Scaffold, Butina clustering, MaxMin, random, time-based
- **Landscape roughness analysis**: NNDR (classification) and SALI (regression), supporting all 4 fingerprint types
- **Conformal prediction & ECE**: uncertainty calibration for both regression and classification
- **Hyperparameter optimization** via Optuna
- **CLI entry points** for single-dataset and batch experiments

---

## Installation

```bash
# Core (classical ML models only: RF, XGBoost, SVM, Ridge, Lasso, Logistic Regression)
pip install envmolbench

# Core + CatBoost
pip install envmolbench[catboost]

# Core + hyperparameter optimization (Optuna)
pip install envmolbench[optuna]

# Core + deep learning models (ChemBERTa, Chemprop, GNN, CNN, UniMol)
# Note: torch-geometric requires a PyTorch-compatible build.
# See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install envmolbench[dl]

# Core + Mordred descriptors featurizer
pip install envmolbench[mordred]

# Everything
pip install envmolbench[all]
```

---

## Dataset Setup

The built-in datasets are **not bundled** in the pip package (too large). You need to provide the dataset files separately.

**Option 1 — Environment variable (recommended for pip install):**
```bash
export ENVMOLBENCH_DATA_DIR=/path/to/your/datasets
```

**Option 2 — Pass `datasets_dir` explicitly:**
```python
smiles, labels, task = eb.load_dataset("esol", datasets_dir="/path/to/datasets")
```

**Option 3 — Development mode (clone the repo):**  
The loader automatically finds `datasets/` next to the package root when running from a cloned repository.

> The dataset files can be downloaded from the project website or GitHub releases.

**Option 4 — Download via API (recommended):**
```python
import envmolbench as eb

# Download a single dataset (saved to ~/.envmolbench/datasets/ by default)
eb.download_dataset("hlm")
eb.download_dataset("hlm", save_dir="/my/data")  # custom directory

# Download all 45 datasets at once
eb.download_all_datasets()
eb.download_all_datasets(save_dir="/my/data", overwrite=False)
```

After downloading, point the loader to the directory:
```bash
export ENVMOLBENCH_DATA_DIR=~/.envmolbench/datasets
```

---

## Quick Start

### Load a built-in dataset

```python
import envmolbench as eb

smiles, labels, task = eb.load_dataset("esol")
print(task)    # "regression"
print(len(smiles))  # number of molecules
```

### Load a local CSV file

```python
# Auto-detects SMILES column (must contain "smi") and label column
smiles, labels, task = eb.load_dataset("my_data.csv")

# Manual column selection
smiles, labels, task = eb.load_dataset(
    "my_data.csv",
    smiles_col="molecule",
    label_col="toxicity",
    task_type="classification",
)
```

### One-liner training

```python
results = eb.quick_train(model="rf", dataset="esol", split="scaffold")
print(results)
# {"train_rmse": 0.41, "val_rmse": 0.67, "test_rmse": 0.72, ...}
```

### Step-by-step pipeline

```python
import envmolbench as eb

smiles, labels, task = eb.load_dataset("hepatotoxicity")

# Split data
splits = eb.split_data(smiles, labels, method="scaffold")

# Get a model
model = eb.get_model("rf", task_type=task)

# Train and predict
model.fit(splits.train_smiles, splits.train_labels,
          splits.val_smiles,   splits.val_labels)
preds = model.predict(splits.test_smiles)

# Evaluate
metrics = eb.calc_metrics(splits.test_labels, preds, task)
print(metrics)
# {"auc": 0.83, "accuracy": 0.76, "f1": 0.74, ...}
```

### Cross-validation

```python
import envmolbench as eb

smiles, labels, task = eb.load_dataset("hepatotoxicity")

# 5-fold scaffold cross-validation
cv_splits = eb.split_data(smiles, labels, method="scaffold", n_folds=5)
for fold in cv_splits:
    model = eb.get_model("rf", task_type=task)
    model.fit(fold.train_smiles, fold.train_labels,
              fold.val_smiles,   fold.val_labels)
    preds = model.predict(fold.test_smiles)
    print(eb.calc_metrics(fold.test_labels, preds, task))
```

### Landscape roughness analysis

```python
import envmolbench as eb

smiles, labels, task = eb.load_dataset("hepatotoxicity")

# Single fingerprint (default: ecfp4)
r = eb.roughness((smiles, labels, task))
print(r)
# {"task_type": "classification", "fp_type": "ecfp4",
#  "metric": "NNDR", "mean": 0.312, "std": 0.015, "n_valid": 450}

# All four fingerprint types at once
r = eb.roughness(
    (smiles, labels, task),
    fp_type=["ecfp4", "maccs", "rdkit_topo", "atompair"],
)
print(r)
# {"ecfp4": {...}, "maccs": {...}, "rdkit_topo": {...}, "atompair": {...}}

# With a local dataset, using separate smiles/labels
r = eb.compute_roughness(smiles, labels, task_type="classification")

# Fine-grained control
nndr_mean, nndr_std, n = eb.compute_nndr(smiles, labels, fp_type="ecfp4",
                                          max_samples=1000, n_repeats=5)
sali_mean, sali_std,  n = eb.compute_sali(smiles, labels, fp_type="ecfp4")
```

**Roughness parameters:**

| Parameter | Default | Description |
|---|---|---|
| `fp_type` | `"ecfp4"` | Fingerprint type: str or list of str |
| `max_samples` | `2000` | Max molecules before subsampling (NNDR) |
| `n_repeats` | `10` | Subsampling repeats for variance estimation |
| `seed` | `42` | Random seed |

### Conformal prediction & uncertainty calibration

```python
import numpy as np
import envmolbench as eb

# --- Classification ---
y_true = np.array([1, 0, 1, 0, 1, 0] * 20)
y_prob = np.array([0.85, 0.15, 0.78, 0.22, 0.91, 0.08] * 20)

result = eb.conformal_prediction(y_true, y_prob, task_type="classification",
                                  nominal_level=0.90, n_repeats=10)
print(result)
# {"task_type": "classification", "nominal_level": 0.9,
#  "coverage_mean": 0.912, "coverage_std": 0.018,
#  "calibration_factor": None, "ece": 0.041, "n_samples": 120}

ece = eb.compute_ece(y_true, y_prob, task_type="classification")

# --- Regression ---
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

# Low-level access
from envmolbench.conformal import conformal_regression, conformal_classification
```

---

## Available Datasets (45)

### Classification — Toxicity & Biological Activity (17)

| Dataset Name | Description |
|---|---|
| `hepatotoxicity` | Hepatotoxicity (liver toxicity) |
| `Hepatotoxicants` | Hepatotoxicant classification |
| `androgen_receptor` | Androgen receptor binding activity |
| `estrogen_receptor` | Estrogen receptor α binding |
| `antibacterial_activity` | Antibacterial activity |
| `ames_mutagenicity` | Ames mutagenicity |
| `carcinogenicity` | Carcinogenicity |
| `oral_carcinogenicity` | Oral carcinogenicity |
| `cytotoxicity` | Cytotoxicity |
| `neurotoxicity` | Neurotoxicity |
| `ocular_toxicity` | Ocular (eye) toxicity |
| `prenatal_development` | Prenatal developmental toxicity |
| `reproductive_toxicity` | Reproductive toxicity |
| `respiratory_toxicity` | Respiratory toxicity |
| `skin_corrosion` | Skin corrosion |
| `pbt` | Persistent, bioaccumulative and toxic (PBT) |
| `tshr_agonist` | TSHR agonist activity |

### Regression — Physical-Chemical & ADME Properties (14)

| Dataset Name | Description |
|---|---|
| `esol` | Aqueous solubility (log mol/L) |
| `freesolv` | Hydration free energy (kcal/mol) |
| `lipophilicity` | Lipophilicity (log D) |
| `solubility` | Thermodynamic solubility |
| `fubrain` | Fraction unbound in brain |
| `fup` | Fraction unbound in plasma |
| `clin_fup` | Clinical fraction unbound in plasma |
| `hlm` | Human liver microsome clearance |
| `oral_bioavailability` | Oral bioavailability |
| `pka_acidic` | Acidic pKa |
| `kh` | Henry's law constant |
| `pparg_ic50` | PPARγ IC50 |
| `pparg_pkd` | PPARγ pKd |
| `pfas` | PFAS environmental property |

### Environmental Chemistry (9)

| Dataset Name | Description |
|---|---|
| `aero_bio_c` | Aerobic biodegradation (classification) |
| `aero_bio_r` | Aerobic biodegradation (regression) |
| `tfishbio` | Fish bioconcentration factor |
| `koa` | Octanol-air partition coefficient |
| `koc` | Organic carbon partition coefficient |
| `koc2` | Organic carbon partition coefficient (alt.) |
| `tbp` | Tributyl phosphate partitioning |
| `si_oh` | OH radical rate constant |
| `si_so4` | SO₄ radical rate constant |

### Other Molecular Properties (5)

| Dataset Name | Description |
|---|---|
| `ccs_mh` | Collision cross section [M+H]⁺ |
| `ccs_mna` | Collision cross section [M+Na]⁺ |
| `fba` | Fractional binding affinity |
| `fbc` | Fractional binding constant |
| `plv` | Partition lung volume |

---

## Available Models (14)

| Model Name | Type | Notes |
|---|---|---|
| `rf` / `random_forest` | Classical ML | core install |
| `xgboost` | Classical ML | core install |
| `catboost` | Classical ML | core install |
| `svc` / `svr` | Classical ML | core install |
| `ridge` | Classical ML | core install |
| `lasso` | Classical ML | core install |
| `logistic_regression` | Classical ML | core install |
| `gnn` | Deep Learning (GNN) | `pip install envmolbench[dl]` |
| `gcn` | Deep Learning (GCN) | `pip install envmolbench[dl]` |
| `chemberta` | Transformer (SMILES) | `pip install envmolbench[dl]` |
| `chemprop` | MPNN | `pip install envmolbench[dl]` |
| `unimol` | 3D conformational | `pip install envmolbench[dl]` |
| `cnn` | ResNet18 on mol images | `pip install envmolbench[dl]` |

```python
print(eb.list_models())
```

---

## Available Featurizers (7)

| Featurizer Name | Description | Notes |
|---|---|---|
| `morgan` | Morgan (ECFP4) binary fingerprint | core |
| `morgan_count` | Morgan count fingerprint | core |
| `maccs` | MACCS 167-bit keys | core |
| `mordred` | ~1800 Mordred descriptors | `pip install envmolbench[mordred]` |
| `graph` | Graph representation for GNNs | `pip install envmolbench[dl]` |
| `image` | Molecular structure image for CNNs | `pip install envmolbench[dl]` |
| `smiles` | Raw SMILES tokenization | `pip install envmolbench[dl]` |

```python
featurizer = eb.get_featurizer("morgan")
X = featurizer.fit_transform(smiles)

print(eb.list_featurizers())
```

---

## Data Splitting Methods (5)

| Method | Key | Description |
|---|---|---|
| Scaffold split | `scaffold` | Murcko scaffold-based (default, recommended) |
| Random split | `random` | Standard stratified random split |
| Butina clustering | `butina` | Similarity threshold clustering |
| MaxMin sampling | `maxmin` | Diversity-based maximally dissimilar split |
| Time-based | `time` | Chronological split (requires time column) |

```python
splits = eb.split_data(smiles, labels, method="scaffold")
# splits.train_smiles, splits.val_smiles, splits.test_smiles
# splits.train_labels, splits.val_labels, splits.test_labels
```

---

## CLI Usage

After installation, two CLI commands are available:

```bash
# Single dataset experiment
envmolbench --model rf --data esol --split scaffold

# Batch experiment across all datasets
envmolbench-batch --model rf,xgboost --split scaffold
```

---

## API Reference

| Function | Description |
|---|---|
| `eb.download_dataset(name, ...)` | Download a single built-in dataset from the website |
| `eb.download_all_datasets(...)` | Download all 45 built-in datasets |
| `eb.load_dataset(name_or_path, ...)` | Load built-in or local CSV dataset |
| `eb.list_datasets()` | List all built-in dataset names |
| `eb.split_data(smiles, labels, method, n_folds)` | Split data into train/val/test (or k-fold CV) |
| `eb.get_model(name, task_type)` | Get a model by name |
| `eb.list_models()` | List all model names |
| `eb.get_featurizer(name)` | Get a featurizer by name |
| `eb.list_featurizers()` | List all featurizer names |
| `eb.calc_metrics(y_true, y_pred, task)` | Compute evaluation metrics |
| `eb.quick_train(model, dataset, split)` | One-liner training pipeline |
| `eb.roughness(dataset, fp_type, ...)` | Compute landscape roughness (NNDR/SALI) |
| `eb.compute_roughness(smiles, labels, ...)` | Roughness from separate arrays |
| `eb.compute_nndr(smiles, labels, fp_type)` | NNDR (classification roughness) |
| `eb.compute_sali(smiles, labels, fp_type)` | SALI (regression roughness) |
| `eb.conformal_prediction(y_true, y_pred, ...)` | Conformal prediction (regression/classification) |
| `eb.compute_ece(y_true, y_pred, ...)` | Expected Calibration Error |

---

## Citation

If you use `envmolbench` in your research, please cite:

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

## License

MIT License. See [LICENSE](LICENSE) for details.
