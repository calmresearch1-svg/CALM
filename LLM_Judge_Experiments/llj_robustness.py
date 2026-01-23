#!/usr/bin/env python3
"""
LLM Judge - Robustness Aspect Evaluator (Async/Concurrent)

This version uses asyncio for concurrent API calls, providing ~5-10x speedup
over the sequential version.

Usage:
    python llj_robustnessV2.py \\
        --original_file JSONs/Robustness/iu_xray_qwen_original_abst_results.jsonl \\
        --cf_file JSONs/Robustness/iu_xray_qwen_cf2_wrong_region_brain_only_abst_results.jsonl \\
        --dataset iu_xray \\
        --judge_model gemini-2.5-flash \\
        --concurrency 10
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from llm_judge import (
    ASPECT_DEFINITIONS, DATASET_EVIDENCE_MODALITY,
    AsyncLLMJudge, EvalTask,
    print_statistics
)
from llm_judge.config import OUTPUTS_ROOT
from llm_judge.data_handlers import get_data_handler
from llm_judge.utils import load_jsonl


# AsyncLLMJudge and EvalTask are now imported from llm_judge package


def generate_output_path(original_file: str, judge_model: str) -> Path:
    """Generate output file path."""
    base_name = Path(original_file).stem
    model_clean = judge_model.replace("/", "-").replace(".", "-")
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{base_name}_LLJ_result_{model_clean}_{date_str}.jsonl"
    
    output_dir = OUTPUTS_ROOT / "Robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir / filename


async def run_async_evaluation(
    original_file: str,
    cf_file: str,
    dataset: str,
    judge_model: str = "gemini-2.5-flash",
    output_file: str = None,
    image_root: str = None,
    start_index: int = 0,
    end_index: int = None,
    limit: int = None,
    concurrency: int = 10,
    dry_run: bool = False,
) -> str:
    """Run async robustness evaluation."""
    # Get configurations
    aspect_config = ASPECT_DEFINITIONS["robustness"]
    evidence_modality = DATASET_EVIDENCE_MODALITY.get(dataset, "medical image")
    
    # Get data handler
    image_root_path = Path(image_root) if image_root else None
    data_handler = get_data_handler(dataset, image_root_path)
    
    # Load data files
    print(f"Loading original results from {original_file}...")
    original_data = load_jsonl(original_file)
    
    print(f"Loading counterfactual results from {cf_file}...")
    cf_data = load_jsonl(cf_file)
    
    # Create maps by entry ID
    original_map = {data_handler.get_entry_id(item): item for item in original_data}
    cf_map = {data_handler.get_entry_id(item): item for item in cf_data}
    
    # Find common IDs
    common_ids = sorted(
        list(set(original_map.keys()) & set(cf_map.keys())),
        key=lambda x: int(x) if x.isdigit() else x
    )
    
    # Apply filtering
    if end_index is None:
        end_index = len(common_ids)
    ids_to_process = common_ids[start_index:end_index]
    if limit:
        ids_to_process = ids_to_process[:limit]
    
    print(f"Found {len(common_ids)} common entries. Processing {len(ids_to_process)}.")
    
    # Build task list
    tasks = []
    for entry_id in ids_to_process:
        orig_item = original_map[entry_id]
        cf_item = cf_map[entry_id]
        
        tasks.append(EvalTask(
            entry_id=entry_id,
            image_path=data_handler.get_full_image_path(orig_item),
            image_path_relative=data_handler.get_image_path(orig_item),
            original_q=data_handler.get_question(orig_item),
            original_a=data_handler.get_response(orig_item),
            cf_q=data_handler.get_question(cf_item),
            cf_a=data_handler.get_response(cf_item),
            model_name=data_handler.get_model_name(orig_item),
        ))
    
    # Setup output file
    if output_file is None:
        output_path = generate_output_path(original_file, judge_model)
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Output will be saved to: {output_path}")
    
    if dry_run:
        print("\n[DRY RUN] Would process:")
        for task in tasks[:3]:
            print(f"  Entry {task.entry_id}: {task.original_q[:60]}...")
        print(f"  ... and {len(tasks) - 3} more entries")
        return str(output_path)
    
    # Initialize async judge and run
    start_time = time.time()
    
    judge = AsyncLLMJudge(
        aspect_config=aspect_config,
        judge_model=judge_model,
        evidence_modality=evidence_modality,
        concurrency=concurrency,
    )
    
    await judge.evaluate_batch(tasks, output_path)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/len(tasks):.2f}s per entry)")
    
    # Print statistics
    print_statistics(str(output_path))
    
    return str(output_path)


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Async LLM Judge for Robustness Evaluation (V2 - Concurrent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fast evaluation with 10 concurrent requests
    python llj_robustnessV2.py \\
        --original_file JSONs/Robustness/iu_xray_qwen_original_abst_results.jsonl \\
        --cf_file JSONs/Robustness/iu_xray_qwen_cf2_wrong_region_brain_only_abst_results.jsonl \\
        --dataset iu_xray \\
        --concurrency 10

    # Conservative with 5 concurrent requests  
    python llj_robustnessV2.py \\
        --original_file JSONs/Robustness/iu_xray_qwen_original_abst_results.jsonl \\
        --cf_file JSONs/Robustness/iu_xray_qwen_cf2_wrong_region_brain_only_abst_results.jsonl \\
        --dataset iu_xray \\
        --concurrency 5
        """
    )
    
    # Required arguments
    parser.add_argument("--original_file", type=str, required=True)
    parser.add_argument("--cf_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["iu_xray", "ham10000", "harvard_fairvlmed"])
    
    # Optional arguments
    parser.add_argument("--judge_model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of concurrent requests (default: 10)")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stats_only", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.stats_only:
        print_statistics(args.stats_only)
        return
    
    evidence_modality = DATASET_EVIDENCE_MODALITY.get(args.dataset, "medical image")
    print(f"\n{'='*60}")
    print("LLM JUDGE - ROBUSTNESS EVALUATION (V2 ASYNC)")
    print(f"{'='*60}")
    print(f"Original file: {args.original_file}")
    print(f"CF file: {args.cf_file}")
    print(f"Dataset: {args.dataset}")
    print(f"Evidence modality: {evidence_modality}")
    print(f"Judge model: {args.judge_model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*60}\n")
    
    # Run async evaluation
    output_file = asyncio.run(run_async_evaluation(
        original_file=args.original_file,
        cf_file=args.cf_file,
        dataset=args.dataset,
        judge_model=args.judge_model,
        output_file=args.output_file,
        image_root=args.image_root,
        start_index=args.start_index,
        end_index=args.end_index,
        limit=args.limit,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    ))
    
    print(f"\nEvaluation complete. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
