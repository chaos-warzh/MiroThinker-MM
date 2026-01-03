#!/usr/bin/env python3
"""
批量运行评估器

Usage:
    # 基本用法 - 评估单个目录
    python run_batch_eval.py single \
        --result-dir result/20251227/datasets_batch2/32k/gpt-4.1 \
        --insights-dir insights/batch2_insights
    
    # 只运行特定指标（跳过 factual_accuracy）
    python run_batch_eval.py single \
        --result-dir result/20251227/datasets_batch2/32k/gpt-4.1 \
        --insights-dir insights/batch2_insights \
        --metrics information_recall overall_quality format_compliance citation_coverage tool_usage
    
    # 批量评估所有模型和上下文大小
    python run_batch_eval.py all \
        --result-base result/20251227-new/datasets_batch2 \
        --output-dir evaluation_logs/batch_20251227_new
    
    # 批量评估，只运行特定指标
    python run_batch_eval.py all \
        --result-base result/20251227-new/datasets_batch2 \
        --metrics information_recall overall_quality
    
    # 批量评估，排除特定模型
    python run_batch_eval.py all \
        --result-base result/20251227-new/datasets_batch2 \
        --exclude-models claude37_sonnet

Available metrics:
    - information_recall: 信息召回率评估
    - factual_accuracy: 事实准确性评估
    - overall_quality: 整体质量评估
    - format_compliance: 格式符合性评估
    - citation_coverage: 引用覆盖率评估
    - tool_usage: 工具使用效率评估
"""

import argparse
import json
import logging
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent))

from evaluators.run_all import EvaluationRunner, ALL_METRICS

# 全局 logger
logger = logging.getLogger("batch_eval")


def setup_logging(output_dir: Path = None, level: int = logging.INFO):
    """
    设置日志系统，同时输出到终端和文件
    
    Args:
        output_dir: 日志文件输出目录，如果为 None 则只输出到终端
        level: 日志级别
    """
    # 创建 logger
    logger.setLevel(level)
    
    # 清除已有的 handlers（避免重复添加）
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 简单格式化器（用于终端，不显示时间戳）
    simple_formatter = logging.Formatter(fmt='%(message)s')
    
    # 添加终端 handler（StreamHandler）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了输出目录，添加文件 handler
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = output_dir / f"evaluation_log_{timestamp}.txt"
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"📝 Logging to: {log_path}")
    
    return logger


def run_batch_evaluation(result_dir: Path, insights_dir: Path, output_dir: Path = None, 
                         checklist_dir: Path = None, datasets_dir: Path = None, context_size: str = "32k",
                         skip_existing: bool = True, metrics: list = None):
    """批量运行评估
    
    Args:
        result_dir: 结果目录，包含各个 case 的子目录
        insights_dir: insights 目录，包含 gold insights
        output_dir: 输出目录
        checklist_dir: checklist 目录
        datasets_dir: 数据集目录，包含 long_context 和 useful_search 文件
        context_size: 上下文大小 (32k, 64k, 128k, 256k)
        skip_existing: 是否跳过已评测的 case（默认 True）
        metrics: 要运行的指标列表，如果为 None 则运行所有指标
    """
    
    # 获取所有 case 目录
    case_dirs = sorted([d for d in result_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    logger.info(f"Found {len(case_dirs)} cases to evaluate")
    logger.info(f"Skip existing: {skip_existing}")
    logger.info(f"Result dir: {result_dir}")
    logger.info(f"Insights dir: {insights_dir}")
    if datasets_dir:
        logger.info(f"Datasets dir: {datasets_dir}")
    if checklist_dir:
        logger.info(f"Checklist dir: {checklist_dir}")
    logger.info(f"Context size: {context_size}")
    if metrics:
        logger.info(f"Metrics: {', '.join(metrics)}")
    else:
        logger.info(f"Metrics: all ({', '.join(ALL_METRICS)})")
    logger.info("=" * 60)
    
    # 创建评估运行器
    runner = EvaluationRunner(metrics=metrics)
    
    # 存储所有结果
    all_results = {}
    summary_scores = []
    
    skipped_existing = 0
    
    for case_dir in case_dirs:
        case_id = case_dir.name
        
        # 检查是否已评测（跳过已存在的结果）
        if skip_existing and output_dir:
            existing_result = output_dir / case_id / "combined_result.json"
            if existing_result.exists():
                logger.info(f"  ⏭️ Skipping case {case_id}: already evaluated")
                skipped_existing += 1
                
                # 加载已有结果用于汇总
                try:
                    existing_data = json.loads(existing_result.read_text(encoding='utf-8'))
                    all_results[case_id] = existing_data
                    summary_scores.append({
                        'case_id': case_id,
                        'total_score': existing_data.get('total_score', 0),
                        'metrics': {name: m.get('score', 0) for name, m in existing_data.get('metrics', {}).items()}
                    })
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to load existing result for case {case_id}: {e}")
                continue
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Evaluating case {case_id}...")
        logger.info("=" * 60)
        
        # 检查必要文件
        final_report = case_dir / "final_report.md"
        execution_log = case_dir / "execution_log.json"
        insights_case_dir = insights_dir / case_id
        
        if not final_report.exists():
            logger.warning(f"  ⚠️ Skipping case {case_id}: final_report.md not found")
            continue
        
        # 准备参数
        gold_path = insights_case_dir / "gold_insights_from_longcontext.json" if insights_case_dir.exists() else None
        gold_source_path = insights_case_dir / "gold_insights_from_source.json" if insights_case_dir.exists() else None
        
        # 查找 long_context 和 useful_search 文件
        long_context_path = None
        useful_search_path = None
        source_folder = None
        if datasets_dir:
            datasets_case_dir = datasets_dir / case_id
            if datasets_case_dir.exists():
                # 根据 context_size 查找对应的 long_context 文件
                long_context_file = datasets_case_dir / f"long_context_sampled_{context_size}.json"
                if long_context_file.exists():
                    long_context_path = long_context_file
                
                # 查找 useful_search 文件
                # 查找 useful_search 文件（支持 .json 和 .jsonl 格式）
                useful_search_file = datasets_case_dir / "useful_search.json"
                useful_search_file_jsonl = datasets_case_dir / "useful_search.jsonl"
                if useful_search_file.exists():
                    useful_search_path = useful_search_file
                elif useful_search_file_jsonl.exists():
                    useful_search_path = useful_search_file_jsonl
                
                # 设置 source_folder
                source_folder = datasets_case_dir
        
        # 查找 checklist 文件
        checklist_path = None
        if checklist_dir:
            checklist_case_dir = checklist_dir / case_id
            if checklist_case_dir.exists():
                checklist_path = checklist_case_dir / "checklist.json"
                if not checklist_path.exists():
                    checklist_path = None
        
        try:
            # 运行评估
            result = runner.run(
                result_path=final_report,
                gold_path=gold_path if gold_path and gold_path.exists() else None,
                gold_source_path=gold_source_path if gold_source_path and gold_source_path.exists() else None,
                execution_log_path=execution_log if execution_log.exists() else None,
                checklist_path=checklist_path,
                long_context_path=long_context_path,
                source_folder=source_folder,
                useful_search_path=useful_search_path,
                case_id=case_id
            )
            
            # 生成报告
            report = runner.generate_report(result)
            logger.info(report)
            
            # 保存结果
            all_results[case_id] = result.to_dict()
            
            # 提取分数，将 information_recall 拆分为两个子指标
            metrics_dict = {}
            for name, r in result.results.items():
                if name == 'information_recall':
                    # 拆分 information_recall 为 long_context 和 source_documents
                    details = r.details
                    components = details.get('components', {})
                    
                    lc = components.get('long_context', {})
                    if lc.get('available'):
                        metrics_dict['information_recall_longcontext'] = lc.get('score', 0)
                    
                    src = components.get('source_documents', {})
                    if src.get('available'):
                        metrics_dict['information_recall_source'] = src.get('score', 0)
                    
                    # 也保留综合分数
                    metrics_dict['information_recall'] = r.score
                else:
                    metrics_dict[name] = r.score
            
            summary_scores.append({
                'case_id': case_id,
                'total_score': result.total_score,
                'metrics': metrics_dict
            })
            
            # 保存单个 case 的结果
            if output_dir:
                case_output_dir = output_dir / case_id
                case_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存综合结果
                (case_output_dir / "combined_result.json").write_text(
                    result.to_json(), encoding='utf-8'
                )
                
                # 保存报告
                (case_output_dir / "evaluation_report.txt").write_text(
                    report, encoding='utf-8'
                )
                
        except Exception as e:
            logger.error(f"  ❌ Error evaluating case {case_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 打印汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 EVALUATION SUMMARY")
    logger.info("=" * 60)
    
    if summary_scores:
        # 计算平均分
        avg_total = sum(s['total_score'] for s in summary_scores) / len(summary_scores)
        
        logger.info(f"\n📈 Average Total Score: {avg_total:.1f}/100")
        if skipped_existing > 0:
            logger.info(f"   (Including {skipped_existing} previously evaluated cases)")
        logger.info(f"\n📋 Individual Scores:")
        
        for s in summary_scores:
            logger.info(f"  Case {s['case_id']}: {s['total_score']:.1f}/100")
            for metric, score in s['metrics'].items():
                logger.info(f"    - {metric}: {score:.1f}")
        
        # 计算各维度平均分
        logger.info(f"\n📊 Average Scores by Dimension:")
        metric_names = list(summary_scores[0]['metrics'].keys())
        for metric in metric_names:
            avg = sum(s['metrics'].get(metric, 0) for s in summary_scores) / len(summary_scores)
            logger.info(f"  - {metric}: {avg:.1f}/100")
        
        # 保存汇总结果
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = output_dir / "summary.json"
            summary_data = {
                'evaluation_time': datetime.now().isoformat(),
                'result_dir': str(result_dir),
                'insights_dir': str(insights_dir),
                'total_cases': len(summary_scores),
                'average_total_score': avg_total,
                'average_by_dimension': {
                    metric: sum(s['metrics'].get(metric, 0) for s in summary_scores) / len(summary_scores)
                    for metric in metric_names
                },
                'individual_scores': summary_scores
            }
            summary_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding='utf-8')
            logger.info(f"\n📁 Summary saved to: {summary_path}")
    else:
        logger.warning("  ⚠️ No cases were successfully evaluated")
    
    return all_results


def discover_models_and_contexts(result_base: Path) -> List[tuple]:
    """
    自动发现 result_base 目录下的所有 (context_size, model) 组合
    
    目录结构应为: result_base/<context_size>/<model>/
    
    Returns:
        List of (context_size, model) tuples
    """
    combinations = []
    
    if not result_base.exists():
        logger.error(f"Error: Result base directory not found: {result_base}")
        return combinations
    
    # 遍历第一层目录 (context_size)
    for context_dir in sorted(result_base.iterdir()):
        if not context_dir.is_dir():
            continue
        
        context_size = context_dir.name
        
        # 遍历第二层目录 (model)
        for model_dir in sorted(context_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            
            model = model_dir.name
            
            # 检查是否有 case 目录 (数字命名的目录)
            case_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if case_dirs:
                combinations.append((context_size, model))
    
    return combinations


def evaluate_single_config(
    config: Tuple[str, str],
    result_base: Path,
    insights_dir: Path,
    datasets_dir: Path,
    checklist_dir: Path,
    output_dir: Path,
    skip_existing: bool,
    metrics: list = None
) -> Tuple[str, str, Dict]:
    """
    评估单个配置（用于并发执行）
    
    Returns:
        (context_size, model, result_dict)
    """
    context_size, model = config
    result_dir = result_base / context_size / model
    eval_output_dir = output_dir / context_size / model
    
    # 检查是否有 case 目录
    case_dirs = [d for d in result_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"[{context_size}/{model}] Starting evaluation...")
    logger.info(f"[{context_size}/{model}] Cases found: {len(case_dirs)}")
    if metrics:
        logger.info(f"[{context_size}/{model}] Metrics: {', '.join(metrics)}")
    logger.info("=" * 60)
    
    try:
        run_batch_evaluation(
            result_dir=result_dir,
            insights_dir=insights_dir,
            output_dir=eval_output_dir,
            checklist_dir=checklist_dir,
            datasets_dir=datasets_dir,
            context_size=context_size,
            skip_existing=skip_existing,
            metrics=metrics
        )
        
        # 读取 summary.json
        summary_path = eval_output_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
                
            result = {
                'total_cases': summary_data.get('total_cases', 0),
                'average_total_score': summary_data.get('average_total_score', 0),
                'average_by_dimension': summary_data.get('average_by_dimension', {})
            }
            logger.info(f"✅ [{context_size}/{model}] Completed! Score: {result['average_total_score']:.1f}")
            return context_size, model, result
        
        logger.warning(f"⚠️ [{context_size}/{model}] No summary.json found")
        return context_size, model, None
        
    except Exception as e:
        logger.error(f"❌ [{context_size}/{model}] Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return context_size, model, None


def run_all_evaluations(
    result_base: Path,
    insights_dir: Path,
    datasets_dir: Path = None,
    checklist_dir: Path = None,
    output_dir: Path = None,
    skip_existing: bool = True,
    max_workers: int = 4,
    metrics: list = None,
    exclude_models: list = None
) -> Dict:
    """
    运行所有发现的模型和上下文大小的评估
    
    Args:
        max_workers: 最大并发数（默认 4）
        metrics: 要运行的指标列表，如果为 None 则运行所有指标
        exclude_models: 要排除的模型列表
    
    Returns:
        Dict with all results
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = Path("evaluation_logs") / result_base.name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置全局日志（在发现配置之前设置，这样所有输出都会被记录）
    setup_logging(output_dir)
    
    # 自动发现所有组合
    combinations = discover_models_and_contexts(result_base)
    
    if not combinations:
        logger.error("No model/context combinations found!")
        return {}
    
    # 过滤掉排除的模型
    if exclude_models:
        original_count = len(combinations)
        combinations = [(cs, m) for cs, m in combinations if m not in exclude_models]
        excluded_count = original_count - len(combinations)
        if excluded_count > 0:
            logger.info(f"⏭️ Excluded {excluded_count} configurations matching models: {', '.join(exclude_models)}")
    
    logger.info("=" * 60)
    logger.info("DISCOVERED CONFIGURATIONS")
    logger.info("=" * 60)
    logger.info(f"Result base: {result_base}")
    logger.info(f"Found {len(combinations)} configurations:")
    for context_size, model in combinations:
        logger.info(f"  - {context_size}/{model}")
    logger.info(f"Max workers: {max_workers}")
    if metrics:
        logger.info(f"Metrics: {', '.join(metrics)}")
    else:
        logger.info(f"Metrics: all ({', '.join(ALL_METRICS)})")
    if exclude_models:
        logger.info(f"Excluded models: {', '.join(exclude_models)}")
    logger.info("=" * 60)
    logger.info("")
    
    # 运行每个配置的评估（并发）
    all_results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(
                evaluate_single_config,
                config,
                result_base,
                insights_dir,
                datasets_dir,
                checklist_dir,
                output_dir,
                skip_existing,
                metrics
            ): config
            for config in combinations
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            config = futures[future]
            try:
                context_size, model, result = future.result()
                if result:
                    if context_size not in all_results:
                        all_results[context_size] = {}
                    all_results[context_size][model] = result
            except Exception as e:
                logger.error(f"❌ Error processing {config}: {e}")
    
    # 生成汇总报告
    generate_summary_report(all_results, output_dir)
    
    return all_results


def generate_summary_report(all_results: Dict, output_dir: Path):
    """生成 Markdown 汇总报告"""
    
    if not all_results:
        logger.warning("No results to summarize")
        return
    
    # 获取所有上下文大小和模型
    context_sizes = sorted(all_results.keys(), key=lambda x: int(x.replace('k', '')))
    models = set()
    for cs in context_sizes:
        models.update(all_results[cs].keys())
    models = sorted(list(models))
    
    # 获取所有维度
    all_dimensions = set()
    for cs in context_sizes:
        for model in models:
            if model in all_results.get(cs, {}):
                all_dimensions.update(all_results[cs][model].get('average_by_dimension', {}).keys())
    all_dimensions = sorted(list(all_dimensions))
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 生成 Markdown 报告
    md_lines = []
    md_lines.append('# Evaluation Summary Report')
    md_lines.append('')
    md_lines.append(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    md_lines.append(f'**Output Directory:** `{output_dir}`')
    md_lines.append('')
    md_lines.append('---')
    md_lines.append('')
    
    # 总分表格
    md_lines.append('## 📊 Average Total Score')
    md_lines.append('')
    header = '| Model |'
    separator = '|-------|'
    for cs in context_sizes:
        header += f' {cs} |'
        separator += '------|'
    md_lines.append(header)
    md_lines.append(separator)
    
    for model in models:
        row = f'| {model} |'
        for cs in context_sizes:
            if cs in all_results and model in all_results[cs]:
                score = all_results[cs][model].get('average_total_score', 0)
                row += f' {score:.1f} |'
            else:
                row += ' N/A |'
        md_lines.append(row)
    md_lines.append('')
    
    # 每个维度的表格
    for dimension in all_dimensions:
        md_lines.append(f'## 📊 {dimension.replace("_", " ").title()}')
        md_lines.append('')
        
        header = '| Model |'
        separator = '|-------|'
        for cs in context_sizes:
            header += f' {cs} |'
            separator += '------|'
        md_lines.append(header)
        md_lines.append(separator)
        
        for model in models:
            row = f'| {model} |'
            for cs in context_sizes:
                if cs in all_results and model in all_results[cs]:
                    score = all_results[cs][model].get('average_by_dimension', {}).get(dimension, 0)
                    row += f' {score:.1f} |'
                else:
                    row += ' N/A |'
            md_lines.append(row)
        md_lines.append('')
    
    # 案例数量表格
    md_lines.append('## 📊 Total Cases Evaluated')
    md_lines.append('')
    header = '| Model |'
    separator = '|-------|'
    for cs in context_sizes:
        header += f' {cs} |'
        separator += '------|'
    md_lines.append(header)
    md_lines.append(separator)
    
    for model in models:
        row = f'| {model} |'
        for cs in context_sizes:
            if cs in all_results and model in all_results[cs]:
                cases = all_results[cs][model].get('total_cases', 0)
                row += f' {cases} |'
            else:
                row += ' N/A |'
        md_lines.append(row)
    md_lines.append('')
    
    # 保存带时间戳的 Markdown 文件
    md_output = output_dir / f'SUMMARY_{timestamp}.md'
    with open(md_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    # 打印到终端
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info('\n'.join(md_lines))
    logger.info("")
    logger.info(f'📁 Markdown saved to: {md_output}')


def main():
    parser = argparse.ArgumentParser(description="Batch run evaluators")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # 单目录评估命令 (原有功能)
    single_parser = subparsers.add_parser('single', help='Evaluate a single result directory')
    single_parser.add_argument("--result-dir", type=str, required=True, 
                        help="Path to result directory containing case folders")
    single_parser.add_argument("--insights-dir", type=str, required=True,
                        help="Path to insights directory containing gold insights")
    single_parser.add_argument("--datasets-dir", type=str,
                        help="Path to datasets directory containing long_context and useful_search files")
    single_parser.add_argument("--context-size", type=str, default="32k",
                        choices=["32k", "64k", "128k", "256k", "512k"],
                        help="Context size for long_context file (default: 32k)")
    single_parser.add_argument("--checklist-dir", type=str,
                        help="Path to checklist directory containing checklist.json files")
    single_parser.add_argument("--output-dir", type=str,
                        help="Output directory for evaluation results")
    single_parser.add_argument("--no-skip", action="store_true",
                        help="Do not skip already evaluated cases (re-evaluate all)")
    single_parser.add_argument("--metrics", "-m", type=str, nargs='+',
                        choices=ALL_METRICS,
                        help=f"Specific metrics to run (default: all). Available: {', '.join(ALL_METRICS)}")
    
    # 批量评估命令 (新功能)
    all_parser = subparsers.add_parser('all', help='Evaluate all models and context sizes')
    all_parser.add_argument("--result-base", "-r", type=str, required=True,
                        help="Base directory containing context_size/model subdirectories")
    all_parser.add_argument("--insights-dir", "-i", type=str, default="insights/batch2_insights",
                        help="Path to insights directory (default: insights/batch2_insights)")
    all_parser.add_argument("--datasets-dir", "-d", type=str, default="datasets_batch2",
                        help="Path to datasets directory (default: datasets_batch2)")
    all_parser.add_argument("--checklist-dir", "-c", type=str, default="checklists/datasets_batch2",
                        help="Path to checklist directory (default: checklists/datasets_batch2)")
    all_parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for evaluation logs (default: evaluation_logs/<result-base-name>)")
    all_parser.add_argument("--no-skip", action="store_true",
                        help="Do not skip already evaluated cases (re-evaluate all)")
    all_parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    all_parser.add_argument("--metrics", "-m", type=str, nargs='+',
                        choices=ALL_METRICS,
                        help=f"Specific metrics to run (default: all). Available: {', '.join(ALL_METRICS)}")
    all_parser.add_argument("--exclude-models", "-e", type=str, nargs='+',
                        help="Models to exclude from evaluation (e.g., claude37_sonnet)")
    
    args = parser.parse_args()
    
    if args.command == 'single':
        # 单目录评估
        result_dir = Path(args.result_dir)
        insights_dir = Path(args.insights_dir)
        datasets_dir = Path(args.datasets_dir) if args.datasets_dir else None
        checklist_dir = Path(args.checklist_dir) if args.checklist_dir else None
        output_dir = Path(args.output_dir) if args.output_dir else result_dir / "eval_results"
        
        # 设置日志
        setup_logging(output_dir)
        
        run_batch_evaluation(
            result_dir=result_dir, 
            insights_dir=insights_dir, 
            output_dir=output_dir, 
            checklist_dir=checklist_dir,
            datasets_dir=datasets_dir,
            context_size=args.context_size,
            skip_existing=not args.no_skip,
            metrics=args.metrics
        )
    
    elif args.command == 'all':
        # 批量评估所有模型和上下文大小
        result_base = Path(args.result_base)
        insights_dir = Path(args.insights_dir)
        datasets_dir = Path(args.datasets_dir) if args.datasets_dir else None
        checklist_dir = Path(args.checklist_dir) if args.checklist_dir else None
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        run_all_evaluations(
            result_base=result_base,
            insights_dir=insights_dir,
            datasets_dir=datasets_dir,
            checklist_dir=checklist_dir,
            output_dir=output_dir,
            skip_existing=not args.no_skip,
            max_workers=args.workers,
            metrics=args.metrics,
            exclude_models=args.exclude_models
        )
    
    else:
        # 没有指定子命令，打印帮助
        parser.print_help()
        print("\n" + "=" * 60)
        print("示例用法:")
        print("=" * 60)
        print("\n# 评估单个目录 (单个模型+单个上下文大小)")
        print("uv run python run_batch_eval.py single \\")
        print("    --result-dir result/20251227-new/datasets_batch2/32k/gpt-4.1 \\")
        print("    --insights-dir insights/batch2_insights")
        print("\n# 评估所有模型和上下文大小 (自动发现)")
        print("uv run python run_batch_eval.py all \\")
        print("    --result-base result/20251227-new/datasets_batch2 \\")
        print("    --output-dir evaluation_logs/batch_20251227_new")


if __name__ == "__main__":
    main()
