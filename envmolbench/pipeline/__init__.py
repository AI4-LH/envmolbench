"""
pipeline 子包：流程编排与超参数优化目标函数。

对外暴露：
    PipelineRunner    —— 标准训练-评估流程编排器
                         run()          : 固定超参数直接训练
                         run_hyperopt() : Optuna 超参数搜索 + 重训

    HyperoptObjective —— Optuna 超参数优化目标函数

    HyperoptObjective 支持：
    - 传统 ML 模型（rf/xgboost/catboost/svm/logistic_regression/ridge/lasso）
    - 深度学习模型（gnn/chemprop/chemberta/cnn/unimol）
    - 静态特征预计算（maccs/mordred，避免每次 trial 重复计算）
    - Mordred 描述符 per-trial 特征筛选（防数据泄漏）
    - XGBoost/CatBoost 早停（单次验证模式）
    - n_cv_folds > 1 时的交叉验证模式
    - 双层超时保护（trial_timeout_seconds + total_timeout_seconds）
"""

from .runner import PipelineRunner
from .objective import HyperoptObjective

__all__ = ["PipelineRunner", "HyperoptObjective"]
