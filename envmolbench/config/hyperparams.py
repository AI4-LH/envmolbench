"""
超参数搜索空间定义（供 Optuna 使用）。

迁移自：cpu_ml_gnn/hyperparams.py

重要设计说明：
    1. Morgan 指纹的 radius 和 length 是超参数（与原版对齐）
    2. SVM/LogisticRegression 的 scaler 类型是超参数
    3. 分类任务的 RF/XGBoost/CatBoost 包含类别不平衡处理参数
    4. XGBoost/CatBoost 的 n_estimators/iterations 在原版中依赖早停，
       此处设置大值（2000），实际迭代由早停控制

使用示例::

    import optuna
    from envmolbench.config.hyperparams import get_search_space

    def objective(trial):
        params = get_search_space("rf", trial, task_type="regression",
                                  featurizer="morgan")
        # ... 用 params 训练模型 ...

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
"""
from __future__ import annotations

from typing import Callable, Dict, Optional


# ────────────────────────────────────────────────────────────────
# 内部辅助
# ────────────────────────────────────────────────────────────────

def _is_clf(task_type: Optional[str]) -> bool:
    """判断是否分类任务。"""
    return str(task_type).lower().startswith("class")


# ────────────────────────────────────────────────────────────────
# 特征化器搜索空间（Morgan 的 radius/length 是可优化超参数）
# ────────────────────────────────────────────────────────────────

def _morgan_featurizer_params(trial, prefix: str = "") -> Dict:
    """
    Morgan 指纹超参数（对应原版 hyperparams.py 中 'radius' 和 'length'）。
    prefix 用于区分多特征化器场景下的参数命名。
    """
    p = prefix
    return {
        "radius": trial.suggest_int(f"{p}radius", 1, 3),
        "n_bits": trial.suggest_categorical(f"{p}n_bits", [512, 1024, 2048]),
    }


# ────────────────────────────────────────────────────────────────
# 传统机器学习模型搜索空间
# ────────────────────────────────────────────────────────────────

def _rf_search_space(trial, task_type: Optional[str] = None,
                     featurizer: Optional[str] = None) -> Dict:
    """随机森林超参数搜索空间。"""
    params = {
        # 树数量：原版 50~1500，此处保守范围（早停不适用于 RF）
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth":         trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features":      trial.suggest_categorical(
                                 "max_features", ["sqrt", "log2", 0.5, 0.7]
                             ),
    }
    # 分类任务：类别不平衡处理（对应原版 class_weight）
    if _is_clf(task_type):
        params["class_weight"] = trial.suggest_categorical(
            "class_weight", ["balanced", None]
        )
    # Morgan 时 radius/length 作为超参数
    if featurizer in ("morgan", "morgan_count"):
        params.update(_morgan_featurizer_params(trial))
    return params


def _xgboost_search_space(trial, task_type: Optional[str] = None,
                           featurizer: Optional[str] = None) -> Dict:
    """XGBoost 超参数搜索空间。"""
    params = {
        # n_estimators 设大值，依赖早停控制实际轮数（对应原版逻辑）
        "n_estimators":     trial.suggest_int("n_estimators", 500, 2000, step=100),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    # 分类任务：类别不平衡处理（scale_pos_weight 对应原版）
    if _is_clf(task_type):
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, 20.0)
    if featurizer in ("morgan", "morgan_count"):
        params.update(_morgan_featurizer_params(trial))
    return params


def _svm_search_space(trial, task_type: Optional[str] = None,
                      featurizer: Optional[str] = None) -> Dict:
    """
    SVM (SVR/SVC) 超参数搜索空间。
    scaler 是 SVM 的重要超参数（原版中 SVC 包含 scaler 参数）。
    """
    params = {
        "C":      trial.suggest_float("C", 0.01, 100.0, log=True),
        "gamma":  trial.suggest_float("gamma", 1e-6, 0.1, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
        # scaler 是 SVM 的必要超参数（SVM 对特征尺度敏感）
        "scaler": trial.suggest_categorical(
            "scaler", ["standard", "minmax", "maxabs", "robust"]
        ),
    }
    if _is_clf(task_type):
        params["class_weight"] = trial.suggest_categorical(
            "class_weight", ["balanced", None]
        )
    else:
        # 回归专用（SVR epsilon）
        params["epsilon"] = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
    if featurizer in ("morgan", "morgan_count"):
        params.update(_morgan_featurizer_params(trial))
    return params


def _logistic_regression_search_space(trial, task_type: Optional[str] = None,
                                       featurizer: Optional[str] = None) -> Dict:
    """
    LogisticRegression 超参数搜索空间。
    原版使用嵌套 hp.choice（solver + penalty 联动），此处简化为 solver_penalty 组合。
    scaler 是 LogisticRegression 的必要超参数。
    """
    # solver-penalty 合法组合（sklearn 的约束）
    solver_penalty = trial.suggest_categorical(
        "solver_penalty",
        ["saga_l1", "saga_l2", "lbfgs_l2", "liblinear_l1", "liblinear_l2"]
    )
    solver, penalty = solver_penalty.split("_", 1)
    params = {
        "C":          trial.suggest_float("C", 0.01, 100.0, log=True),
        "max_iter":   trial.suggest_int("max_iter", 10000, 30000, step=1000),
        "solver":     solver,
        "penalty":    penalty,
        "class_weight": trial.suggest_categorical(
            "class_weight", ["balanced", None]
        ),
        # scaler 是必要超参数
        "scaler":     trial.suggest_categorical(
            "scaler", ["standard", "minmax", "maxabs", "robust"]
        ),
    }
    if featurizer in ("morgan", "morgan_count"):
        params.update(_morgan_featurizer_params(trial))
    return params


def _catboost_search_space(trial, task_type: Optional[str] = None,
                            featurizer: Optional[str] = None) -> Dict:
    """CatBoost 超参数搜索空间。"""
    params = {
        "iterations":          trial.suggest_int("iterations", 500, 2000, step=100),
        "depth":               trial.suggest_int("depth", 3, 10),
        "learning_rate":       trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 3.0, 30.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
    }
    # 分类任务：自动类别权重（对应原版 auto_class_weights）
    if _is_clf(task_type):
        params["auto_class_weights"] = trial.suggest_categorical(
            "auto_class_weights", ["Balanced", "SqrtBalanced", None]
        )
    if featurizer in ("morgan", "morgan_count"):
        params.update(_morgan_featurizer_params(trial))
    return params


def _ridge_lasso_search_space(trial, task_type: Optional[str] = None,
                               featurizer: Optional[str] = None) -> Dict:
    """Ridge / Lasso 超参数搜索空间（线性模型，缩放已在 ClassicalModel 中自动处理）。"""
    params = {
        "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
    }
    if featurizer in ("morgan", "morgan_count"):
        params.update(_morgan_featurizer_params(trial))
    return params


def _lgbm_search_space(trial, task_type: Optional[str] = None,
                        featurizer: Optional[str] = None) -> Dict:
    """LightGBM 超参数搜索空间。"""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 2000, step=100),
        "max_depth":         trial.suggest_int("max_depth", -1, 15),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    if _is_clf(task_type):
        params["class_weight"] = trial.suggest_categorical(
            "class_weight", ["balanced", None]
        )
    if featurizer in ("morgan", "morgan_count"):
        params.update(_morgan_featurizer_params(trial))
    return params


# ────────────────────────────────────────────────────────────────
# 深度学习模型搜索空间
# ────────────────────────────────────────────────────────────────

def _gnn_search_space(trial, task_type: Optional[str] = None,
                      featurizer: Optional[str] = None) -> Dict:
    """GNN / GCN 超参数搜索空间（对应原版 gnn_suite/objective.py 的 params）。"""
    return {
        "hidden_dim":    trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
        "num_layers":    trial.suggest_int("num_layers", 2, 5),
        "dropout":       trial.suggest_float("drop_rate", 0.0, 0.5),
        "lr":            trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def _chemprop_search_space(trial, task_type: Optional[str] = None,
                            featurizer: Optional[str] = None) -> Dict:
    """Chemprop MPNN 超参数搜索空间。"""
    return {
        "hidden_dim":  trial.suggest_categorical("hidden_dim", [100, 200, 300, 400]),
        "depth":       trial.suggest_int("depth", 2, 6),
        "dropout":     trial.suggest_float("dropout", 0.0, 0.4),
        "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size":  trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def _chemberta_search_space(trial, task_type: Optional[str] = None,
                             featurizer: Optional[str] = None) -> Dict:
    """ChemBERTa 超参数搜索空间。"""
    return {
        "lr":           trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [8, 16, 32]),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }


def _cnn_search_space(trial, task_type: Optional[str] = None,
                      featurizer: Optional[str] = None) -> Dict:
    """CNN（图像模型）超参数搜索空间。"""
    return {
        "arch":       trial.suggest_categorical(
                          "arch", ["resnet18", "resnet34", "efficientnet_b0"]
                      ),
        "lr":         trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "img_size":   trial.suggest_categorical("img_size", [128, 224]),
    }


def _unimol_search_space(trial, task_type: Optional[str] = None,
                          featurizer: Optional[str] = None) -> Dict:
    """Uni-Mol 超参数搜索空间（受限于显存，范围较小）。"""
    return {
        "lr":            trial.suggest_float("lr", 1e-5, 1e-4, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [8, 16, 32]),
        "freeze_layers": trial.suggest_int("freeze_layers", 0, 8),
    }


# ────────────────────────────────────────────────────────────────
# 注册表：模型名称 → 搜索空间函数
# ────────────────────────────────────────────────────────────────

# 所有搜索空间函数签名统一为 (trial, task_type, featurizer) -> Dict
_SEARCH_SPACES: Dict[str, Callable] = {
    # 传统 ML
    "rf":                   _rf_search_space,
    "random_forest":        _rf_search_space,
    "xgboost":              _xgboost_search_space,
    "xgb":                  _xgboost_search_space,
    "svm":                  _svm_search_space,
    "svr":                  _svm_search_space,
    "svc":                  _svm_search_space,
    "logistic_regression":  _logistic_regression_search_space,
    "logreg":               _logistic_regression_search_space,
    "ridge":                _ridge_lasso_search_space,
    "lasso":                _ridge_lasso_search_space,
    "catboost":             _catboost_search_space,
    "lgbm":                 _lgbm_search_space,
    "lightgbm":             _lgbm_search_space,
    # 深度学习
    "gnn":                  _gnn_search_space,
    "gcn":                  _gnn_search_space,
    "chemprop":             _chemprop_search_space,
    "chemberta":            _chemberta_search_space,
    "cnn":                  _cnn_search_space,
    "unimol":               _unimol_search_space,
}


def get_search_space(
    model_name: str,
    trial,
    task_type: Optional[str] = None,
    featurizer: Optional[str] = None,
) -> Dict:
    """
    获取指定模型的 Optuna 超参数采样结果。

    Args:
        model_name:  模型名称（不区分大小写）。
        trial:       optuna.Trial 对象。
        task_type:   任务类型（'regression'/'classification'），影响分类专用参数。
        featurizer:  特征化器名称（'morgan'/'mordred' 等），影响 Morgan 参数。

    Returns:
        超参数字典。

    Raises:
        KeyError: 未知模型名称。
    """
    key = model_name.lower()
    if key not in _SEARCH_SPACES:
        available = sorted(set(_SEARCH_SPACES.keys()))
        raise KeyError(f"未知模型 '{model_name}'，可用：{available}")
    return _SEARCH_SPACES[key](trial, task_type=task_type, featurizer=featurizer)


def list_search_spaces() -> list:
    """返回所有已注册搜索空间的模型名称列表（去重，按字母序）。"""
    seen_funcs = set()
    result = []
    for name, func in _SEARCH_SPACES.items():
        if func not in seen_funcs:
            seen_funcs.add(func)
            result.append(name)
    return sorted(result)
