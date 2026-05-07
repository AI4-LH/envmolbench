"""
批量运行入口：遍历所有内置数据集，支持断点续跑。

用法示例：

    # 单模型 × 所有数据集
    python envmolbench/run_batch.py --model chemprop

    # 多模型 × 所有数据集
    python envmolbench/run_batch.py --model chemprop,rf,gnn

    # 所有模型 × 所有数据集
    python envmolbench/run_batch.py --model all

    # 指定数据集子集（逗号分隔）
    python envmolbench/run_batch.py --model chemprop --datasets esol,freesolv,ames

    # 断点续跑（已完成的跳过）
    python envmolbench/run_batch.py --model chemprop --result results/chemprop.csv --resume

    # 强制重跑（忽略已有结果）
    python envmolbench/run_batch.py --model chemprop --result results/chemprop.csv --no-resume

    # 限制并发（多进程时）
    python envmolbench/run_batch.py --model rf --workers 4
"""
import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

# 确保包根目录在 sys.path 中（直接运行时）
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from envmolbench.common.logger import get_logger
from envmolbench.common.config_loader import load_config
from envmolbench.common.result_writer import is_experiment_done
from envmolbench.data import list_datasets
from envmolbench.models import list_models
from envmolbench.pipeline.runner import PipelineRunner


logger = get_logger("envmolbench.batch")


# ────────────────────────────────────────────────────────────────
# 单任务执行函数（多进程友好，无 self/logger 捕获）
# ────────────────────────────────────────────────────────────────

def _run_one_task(task_args: Tuple) -> Tuple[str, str, Optional[dict], Optional[str]]:
    """
    执行单个（模型, 数据集）组合。
    设计为顶层函数，确保 ProcessPoolExecutor 可序列化。

    Args:
        task_args: (model_name, dataset_name, split, task, datasets_dir,
                    result_csv, resume, config_overrides)

    Returns:
        (model_name, dataset_name, results_dict_or_None, error_msg_or_None)
    """
    (model_name, dataset_name, split, task, datasets_dir,
     result_csv, resume, config_overrides) = task_args

    # 子进程需要重新设置日志
    _logger = get_logger(f"envmolbench.batch.worker.{model_name}")

    # 断点续跑检查
    if resume and result_csv:
        result_path = Path(result_csv)
        if result_path.exists() and is_experiment_done(result_path, model_name, dataset_name, split):
            _logger.info(f"[跳过] {model_name} × {dataset_name} 已完成")
            return model_name, dataset_name, None, "SKIPPED"

    try:
        config = load_config(model_name, extra=config_overrides)
        runner = PipelineRunner(config=config, result_csv=result_csv)
        results = runner.run(
            model_name=model_name,
            dataset_name=dataset_name,
            split_method=split,
            task_type=task,
            datasets_dir=datasets_dir,
        )
        _logger.info(f"[完成] {model_name} × {dataset_name}")
        return model_name, dataset_name, results, None

    except Exception as exc:
        _logger.error(f"[失败] {model_name} × {dataset_name}：{exc}")
        return model_name, dataset_name, None, str(exc)


# ────────────────────────────────────────────────────────────────
# 主批量运行器
# ────────────────────────────────────────────────────────────────

class BatchRunner:
    """
    批量实验运行器。

    Args:
        models:       模型名称列表。
        datasets:     数据集名称列表；None 时使用所有内置数据集。
        split:        数据划分方法，默认 'scaffold'。
        task:         任务类型；None 时自动检测。
        datasets_dir: 数据集目录。
        result_csv:   结果汇总 CSV 路径。
        resume:       是否跳过已完成的实验（断点续跑）。
        workers:      并行工作进程数；1 表示串行执行。
        config_overrides: 额外配置覆盖字典。
    """

    def __init__(
        self,
        models: List[str],
        datasets: Optional[List[str]] = None,
        split: str = "scaffold",
        task: Optional[str] = None,
        datasets_dir: Optional[str] = None,
        result_csv: Optional[str] = None,
        resume: bool = True,
        workers: int = 1,
        config_overrides: Optional[dict] = None,
    ):
        self.models = models
        self.datasets = datasets or list_datasets()
        self.split = split
        self.task = task
        self.datasets_dir = datasets_dir
        self.result_csv = result_csv
        self.resume = resume
        self.workers = max(1, workers)
        self.config_overrides = config_overrides or {}

    def run(self) -> dict:
        """
        执行批量实验，返回汇总统计。

        Returns:
            包含成功/失败/跳过数量及失败详情的字典。
        """
        # 构建所有任务列表
        tasks = []
        for model_name in self.models:
            for dataset_name in self.datasets:
                tasks.append((
                    model_name, dataset_name, self.split, self.task,
                    self.datasets_dir, self.result_csv, self.resume,
                    self.config_overrides,
                ))

        total = len(tasks)
        logger.info(
            f"批量运行启动：{len(self.models)} 个模型 × {len(self.datasets)} 个数据集 "
            f"= {total} 个任务（workers={self.workers}, resume={self.resume}）"
        )

        stats = {"total": total, "success": 0, "skipped": 0, "failed": 0, "failures": []}
        start_time = time.time()

        if self.workers == 1:
            # 串行执行（调试友好）
            for i, task_args in enumerate(tasks, 1):
                model_name, dataset_name = task_args[0], task_args[1]
                logger.info(f"[{i}/{total}] {model_name} × {dataset_name}")
                _, _, results, error = _run_one_task(task_args)
                self._update_stats(stats, model_name, dataset_name, results, error)
                self._log_progress(stats, i, total, start_time)
        else:
            # 并行执行
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                future_to_task = {
                    executor.submit(_run_one_task, task_args): task_args
                    for task_args in tasks
                }
                completed = 0
                for future in as_completed(future_to_task):
                    completed += 1
                    model_name, dataset_name, results, error = future.result()
                    self._update_stats(stats, model_name, dataset_name, results, error)
                    logger.info(f"[{completed}/{total}] 完成：{model_name} × {dataset_name}")
                    self._log_progress(stats, completed, total, start_time)

        # 最终汇总
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"批量运行完成！总耗时：{elapsed:.1f}s")
        logger.info(
            f"  成功：{stats['success']}  |  跳过：{stats['skipped']}  |  失败：{stats['failed']}"
        )
        if stats["failures"]:
            logger.warning("失败任务列表：")
            for f in stats["failures"]:
                logger.warning(f"  {f['model']} × {f['dataset']}：{f['error']}")
        if self.result_csv:
            logger.info(f"结果已保存至：{self.result_csv}")
        logger.info("=" * 60)

        return stats

    @staticmethod
    def _update_stats(stats: dict, model_name: str, dataset_name: str,
                      results: Optional[dict], error: Optional[str]) -> None:
        """更新统计计数。"""
        if error == "SKIPPED":
            stats["skipped"] += 1
        elif error is None:
            stats["success"] += 1
        else:
            stats["failed"] += 1
            stats["failures"].append({
                "model": model_name,
                "dataset": dataset_name,
                "error": error,
            })

    @staticmethod
    def _log_progress(stats: dict, completed: int, total: int, start_time: float) -> None:
        """打印进度和预计剩余时间。"""
        elapsed = time.time() - start_time
        done = stats["success"] + stats["skipped"] + stats["failed"]
        if done > 0 and total > 0:
            avg_time = elapsed / done
            remaining = avg_time * (total - done)
            logger.info(
                f"  进度 {done}/{total} ({100 * done / total:.1f}%)  "
                f"用时 {elapsed:.0f}s  预计剩余 {remaining:.0f}s"
            )


# ────────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="envmolbench-batch",
        description="分子属性预测基准测试框架 —— 批量训练/评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 主要参数 ──────────────────────────────────────────────────
    parser.add_argument(
        "--model", "-m",
        required=True,
        metavar="MODEL[,MODEL...]",
        help="模型名称，多个用逗号分隔；'all' 表示所有内置模型",
    )
    parser.add_argument(
        "--datasets",
        metavar="DS[,DS...]",
        default=None,
        help="数据集名称，多个用逗号分隔；默认使用所有内置47个数据集",
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
        help="数据集目录（默认：envmolbench/datasets/）",
    )
    parser.add_argument(
        "--result", "-r",
        metavar="CSV",
        default=None,
        help="结果保存路径（如 results/batch_results.csv）",
    )

    # ── 断点续跑 ───────────────────────────────────────────────────
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="跳过已完成的实验（默认启用）",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="强制重跑所有实验（忽略已有结果）",
    )

    # ── 并行控制 ───────────────────────────────────────────────────
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        metavar="N",
        help="并行工作进程数（默认：1，即串行）",
    )

    # ── 日志控制 ───────────────────────────────────────────────────
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出 DEBUG 级别日志",
    )
    parser.add_argument(
        "--log-file",
        metavar="FILE",
        default=None,
        help="日志文件路径",
    )

    # ── 信息查询 ───────────────────────────────────────────────────
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出所有可用模型并退出",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="列出所有内置数据集并退出",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 设置日志级别
    level = logging.DEBUG if args.verbose else logging.INFO
    global logger
    logger = get_logger("envmolbench.batch", log_file=args.log_file, level=level)

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

    # ── 解析模型列表 ───────────────────────────────────────────────
    if args.model.strip().lower() == "all":
        model_names = list_models()
        logger.info(f"使用所有内置模型（共 {len(model_names)} 个）")
    else:
        model_names = [m.strip() for m in args.model.split(",") if m.strip()]

    # ── 解析数据集列表 ─────────────────────────────────────────────
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        dataset_names = list_datasets()
        logger.info(f"使用所有内置数据集（共 {len(dataset_names)} 个）")

    # ── 执行批量运行 ───────────────────────────────────────────────
    runner = BatchRunner(
        models=model_names,
        datasets=dataset_names,
        split=args.split,
        task=args.task,
        datasets_dir=args.datasets_dir,
        result_csv=args.result,
        resume=args.resume,
        workers=args.workers,
    )
    stats = runner.run()

    # 返回码：有失败则返回 1
    sys.exit(1 if stats["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
