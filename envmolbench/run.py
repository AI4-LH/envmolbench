"""
单数据集 CLI 入口。

用法示例：

    # 单模型单数据集（固定超参数）
    python envmolbench/run.py --model chemprop --data esol

    # 使用 Optuna 超参数搜索（传统 ML 必须指定 --featurizer）
    python envmolbench/run.py --model rf --data esol --hyperopt --featurizer morgan
    python envmolbench/run.py --model xgboost --data esol --hyperopt --featurizer mordred

    # 多模型对比（逗号分隔）
    python envmolbench/run.py --model chemprop,rf,gnn --data esol

    # 自定义 CSV 文件
    python envmolbench/run.py --model chemberta --data ./my_data.csv --split random

    # 覆盖配置参数
    python envmolbench/run.py --model chemprop --data esol --epochs 30 --lr 0.001

    # 超参数搜索覆盖搜索轮数和超时时间
    python envmolbench/run.py --model rf --data esol --hyperopt --featurizer morgan \\
        --n-trials 100 --trial-timeout 300 --total-timeout 3600

    # 保存结果到 CSV
    python envmolbench/run.py --model rf --data esol --result results/output.csv

    # 查看可用模型/数据集
    python envmolbench/run.py --list-models
    python envmolbench/run.py --list-datasets
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# 确保包根目录在 sys.path 中（直接运行时）
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from envmolbench.common.logger import get_logger
from envmolbench.common.config_loader import load_config
from envmolbench.pipeline.runner import PipelineRunner
from envmolbench.models import list_models
from envmolbench.data import list_datasets


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="envmolbench",
        description="分子属性预测基准测试框架 —— 单数据集训练/评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 主要参数 ──────────────────────────────────────────────────
    parser.add_argument(
        "--model", "-m",
        metavar="MODEL[,MODEL...]",
        help="模型名称，多个模型用逗号分隔（如 chemprop,rf,gnn）",
    )
    parser.add_argument(
        "--data", "-d",
        metavar="DATASET",
        help="数据集名称（内置47个）或 CSV 文件路径",
    )
    parser.add_argument(
        "--split", "-s",
        default="scaffold",
        choices=["scaffold", "random", "time", "butina", "maxmin"],
        help="数据划分方法（默认：scaffold）",
    )
    parser.add_argument(
        "--task", "-t",
        choices=["regression", "classification"],
        default=None,
        help="任务类型（默认：自动检测）",
    )
    parser.add_argument(
        "--datasets-dir",
        metavar="DIR",
        default=None,
        help="数据集所在目录（默认：envmolbench/datasets/）",
    )
    parser.add_argument(
        "--result",
        metavar="CSV",
        default=None,
        help="汇总结果 CSV 保存路径（如 results/output.csv）",
    )
    parser.add_argument(
        "--results-dir",
        metavar="DIR",
        default=None,
        help="输出根目录，用于保存预测文件和超参数迭代日志（如 results/）",
    )

    # ── 超参数搜索（Optuna） ────────────────────────────────────────
    hyperopt_group = parser.add_argument_group("超参数搜索（Optuna）")
    hyperopt_group.add_argument(
        "--hyperopt",
        action="store_true",
        help="启用 Optuna 超参数搜索（传统 ML 模型需同时指定 --featurizer）",
    )
    hyperopt_group.add_argument(
        "--featurizer", "-f",
        metavar="FEATURIZER",
        default=None,
        help="特征化器名称（morgan/maccs/mordred/graph/image/smiles），传统 ML 超参数搜索时必须指定",
    )
    hyperopt_group.add_argument(
        "--n-cv-folds",
        type=int,
        default=1,
        metavar="N",
        help="交叉验证折数（默认：1，即单次验证；>1 时启用 CV）",
    )
    hyperopt_group.add_argument(
        "--n-trials",
        type=int,
        default=None,
        metavar="N",
        help="最大 trial 次数（默认：读取 config/base.yaml hyperopt.max_trials=300）",
    )
    hyperopt_group.add_argument(
        "--trial-timeout",
        type=int,
        default=None,
        metavar="SECONDS",
        help="单次 trial 最大允许秒数（默认：读取 config/base.yaml hyperopt.trial_timeout_seconds=600）",
    )
    hyperopt_group.add_argument(
        "--total-timeout",
        type=int,
        default=None,
        metavar="SECONDS",
        help="整体搜索最大允许秒数（默认：读取 config/base.yaml hyperopt.total_timeout_seconds=7200）",
    )

    # ── 超参数覆盖 ─────────────────────────────────────────────────
    override_group = parser.add_argument_group("超参数覆盖（会覆盖 YAML 配置）")
    override_group.add_argument("--epochs",   type=int,   default=None, help="最大训练轮数")
    override_group.add_argument("--lr",       type=float, default=None, help="学习率")
    override_group.add_argument("--batch-size",type=int,  default=None, help="批大小")
    override_group.add_argument("--seed",     type=int,   default=None, help="随机种子（默认：42）")
    override_group.add_argument("--patience", type=int,   default=None, help="早停耐心值")
    override_group.add_argument(
        "--config",
        metavar="JSON",
        default=None,
        help='任意额外配置，JSON 字符串（如 \'{"hidden_dim": 256}\'）',
    )

    # ── 信息查询 ──────────────────────────────────────────────────
    info_group = parser.add_argument_group("信息查询")
    info_group.add_argument(
        "--list-models", action="store_true",
        help="列出所有可用模型并退出",
    )
    info_group.add_argument(
        "--list-datasets", action="store_true",
        help="列出所有内置数据集并退出",
    )

    # ── 日志控制 ──────────────────────────────────────────────────
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="输出 DEBUG 级别日志",
    )
    parser.add_argument(
        "--log-file",
        metavar="FILE",
        default=None,
        help="日志文件路径（同时写文件和终端）",
    )

    return parser


def _build_overrides(args: argparse.Namespace) -> dict:
    """将命令行参数转换为配置覆盖字典。"""
    overrides = {}

    # 训练通用参数
    training_overrides = {}
    if args.seed is not None:
        training_overrides["seed"] = args.seed
    if args.patience is not None:
        training_overrides["patience"] = args.patience
    if training_overrides:
        overrides["training"] = training_overrides

    # 通用模型参数（由各模型 YAML 子节点读取）
    model_overrides = {}
    if args.epochs is not None:
        model_overrides["max_epochs"] = args.epochs
    if args.lr is not None:
        model_overrides["lr"] = args.lr
    if args.batch_size is not None:
        model_overrides["batch_size"] = args.batch_size

    # 如果指定了 --config JSON，合并进去
    if args.config:
        try:
            extra = json.loads(args.config)
            model_overrides.update(extra)
        except json.JSONDecodeError as e:
            print(f"[错误] --config 参数不是合法的 JSON：{e}", file=sys.stderr)
            sys.exit(1)

    if model_overrides:
        # 覆盖放在 model_overrides 键下，config_loader 会根据模型名合并
        overrides["_model_overrides"] = model_overrides

    return overrides


def run_single(
    model_name: str,
    dataset: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict:
    """运行单个模型-数据集组合（直接训练或超参数搜索）。"""
    overrides = _build_overrides(args)
    config = load_config(model_name, extra=overrides)

    # results_dir：用于组织预测文件和超参数迭代日志（对应原版输出结构）
    results_dir = getattr(args, "results_dir", None)

    runner = PipelineRunner(
        config=config,
        result_csv=args.result,
        results_dir=results_dir,
    )

    logger.info(f"{'='*60}")
    mode = "超参数搜索（Optuna）" if getattr(args, "hyperopt", False) else "直接训练"
    logger.info(f"模型：{model_name}  |  数据集：{dataset}  |  划分：{args.split}  |  模式：{mode}")
    logger.info(f"{'='*60}")

    if getattr(args, "hyperopt", False):
        # ── 超参数搜索模式 ──────────────────────────────────────
        results = runner.run_hyperopt(
            model_name=model_name,
            dataset_name=dataset,
            featurizer=getattr(args, "featurizer", None),
            n_cv_folds=getattr(args, "n_cv_folds", 1),
            split_method=args.split,
            task_type=args.task,
            datasets_dir=getattr(args, "datasets_dir", None),
            n_trials=getattr(args, "n_trials", None),
            trial_timeout_seconds=getattr(args, "trial_timeout", None),
            total_timeout_seconds=getattr(args, "total_timeout", None),
        )
    else:
        # ── 直接训练模式 ────────────────────────────────────────
        results = runner.run(
            model_name=model_name,
            dataset_name=dataset,
            featurizer=getattr(args, "featurizer", None),
            split_method=args.split,
            task_type=args.task,
            datasets_dir=getattr(args, "datasets_dir", None),
        )
    return results


def _print_results(results: dict, logger: logging.Logger) -> None:
    """格式化打印结果字典。"""
    logger.info("─" * 40)
    logger.info("评估结果：")
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"  {k:<25} {v:.6f}")
        else:
            logger.info(f"  {k:<25} {v}")
    logger.info("─" * 40)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 设置日志
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = get_logger("envmolbench.cli", log_file=args.log_file, level=level)

    # ── 信息查询模式 ───────────────────────────────────────────────
    if args.list_models:
        print("可用模型：")
        for m in list_models():
            print(f"  {m}")
        sys.exit(0)

    if args.list_datasets:
        ds_list = list_datasets()
        print(f"内置数据集（共 {len(ds_list)} 个）：")
        for ds in ds_list:
            print(f"  {ds}")
        sys.exit(0)

    # ── 参数校验 ──────────────────────────────────────────────────
    if not args.model:
        parser.error("请指定 --model 参数（如 --model chemprop）")
    if not args.data:
        parser.error("请指定 --data 参数（如 --data esol）")

    # 解析多模型（逗号分隔）
    model_names = [m.strip() for m in args.model.split(",") if m.strip()]

    # ── 执行训练 ──────────────────────────────────────────────────
    all_results = {}
    for model_name in model_names:
        try:
            results = run_single(model_name, args.data, args, logger)
            _print_results(results, logger)
            all_results[model_name] = results
        except Exception as exc:
            logger.error(f"模型 '{model_name}' 运行失败：{exc}", exc_info=args.verbose)

    # ── 多模型对比汇总 ────────────────────────────────────────────
    if len(model_names) > 1:
        logger.info("\n" + "="*60)
        logger.info("多模型对比汇总")
        logger.info("="*60)
        # 找出共同指标（以测试集指标为主）
        test_keys = set()
        for res in all_results.values():
            test_keys |= {k for k in res if k.startswith("test_")}

        header = f"{'模型':<15}" + "".join(f"{k:<18}" for k in sorted(test_keys))
        logger.info(header)
        logger.info("-" * len(header))
        for model_name, res in all_results.items():
            row = f"{model_name:<15}"
            for k in sorted(test_keys):
                v = res.get(k, "N/A")
                row += f"{v:<18.6f}" if isinstance(v, float) else f"{str(v):<18}"
            logger.info(row)


if __name__ == "__main__":
    main()
