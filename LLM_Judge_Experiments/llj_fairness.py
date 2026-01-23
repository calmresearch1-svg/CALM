#!/usr/bin/env python3
"""
LLM Judge - Fairness Evaluation Script

Evaluates fairness by comparing how a model responds to questions
with different demographic attributes (gender, age, race) for the same image.

Usage:
    python llj_fairness.py \
        --baseline_file JSONs/Fairness/baseline.jsonl \
        --cf_file JSONs/Fairness/counterfactual.jsonl \
        --dataset harvard_fairvlmed \
        --judge_model gemini-2.5-flash
"""

import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

from llm_judge import (
    ASPECT_DEFINITIONS,
    DATASET_EVIDENCE_MODALITY,
    get_data_handler,
)
from llm_judge.utils import load_jsonl
from llm_judge.base_judge import AsyncLLMJudge, EvalTask

# Fixed outputs directory
OUTPUTS_ROOT = Path(__file__).parent / "LLJ_Outputs"

# Demographic types to evaluate
DEMOGRAPHIC_TYPES = ["gender", "age", "race"]


def extract_demographic_type(condition: str) -> Optional[str]:
    """Extract demographic type from condition string."""
    for demo_type in DEMOGRAPHIC_TYPES:
        if demo_type in condition:
            return demo_type
    return None


def generate_output_path(baseline_file: str, judge_model: str, demo_type: str = None) -> Path:
    """Generate output file path."""
    base_name = Path(baseline_file).stem
    model_clean = judge_model.replace("/", "-").replace(".", "-")
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    if demo_type:
        filename = f"{base_name}_{demo_type}_LLJ_result_{model_clean}_{date_str}.jsonl"
    else:
        filename = f"{base_name}_LLJ_result_{model_clean}_{date_str}.jsonl"
    
    output_dir = OUTPUTS_ROOT / "Fairness"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir / filename


def load_completed_entry_ids(output_file: Path) -> set:
    """
    Load entry IDs that have already been processed from an existing output file.
    
    Args:
        output_file: Path to the existing output JSONL file
        
    Returns:
        Set of entry_id strings that have already been processed
    """
    completed_ids = set()
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        entry_id = entry.get("entry_id")
                        if entry_id:
                            completed_ids.add(entry_id)
        except Exception as e:
            print(f"Warning: Error reading existing output file: {e}")
    return completed_ids


def build_fairness_pairs(
    baseline_data: List[Dict],
    cf_data: List[Dict],
    data_handler,
    demographic_filter: Optional[str] = None,
) -> Dict[str, List[Tuple[Dict, Dict]]]:
    """
    Build pairs of baseline and counterfactual entries by (question_id, demographic_type).
    
    Returns:
        Dict mapping demographic_type -> list of (baseline, counterfactual) pairs
    """
    # Index data by (question_id, demographic_type)
    baseline_map: Dict[Tuple[str, str], Dict] = {}
    cf_map: Dict[Tuple[str, str], Dict] = {}
    
    for item in baseline_data:
        entry_id = data_handler.get_entry_id(item)
        condition = item.get("condition", "")
        demo_type = extract_demographic_type(condition)
        if demo_type:
            if demographic_filter is None or demo_type == demographic_filter:
                baseline_map[(entry_id, demo_type)] = item
    
    for item in cf_data:
        entry_id = data_handler.get_entry_id(item)
        condition = item.get("condition", "")
        demo_type = extract_demographic_type(condition)
        if demo_type:
            if demographic_filter is None or demo_type == demographic_filter:
                cf_map[(entry_id, demo_type)] = item
    
    # Find matching pairs
    pairs_by_demo: Dict[str, List[Tuple[Dict, Dict]]] = {t: [] for t in DEMOGRAPHIC_TYPES}
    
    common_keys = set(baseline_map.keys()) & set(cf_map.keys())
    
    for key in sorted(common_keys, key=lambda x: (x[1], str(x[0]))):
        entry_id, demo_type = key
        pairs_by_demo[demo_type].append((baseline_map[key], cf_map[key]))
    
    return pairs_by_demo


async def run_async_evaluation(
    baseline_file: str,
    cf_file: str,
    dataset: str,
    judge_model: str = "gemini-2.5-flash",
    output_file: str = None,
    image_root: str = None,
    demographic_type: str = None,
    start_index: int = 0,
    end_index: int = None,
    limit: int = None,
    concurrency: int = 10,
    dry_run: bool = False,
    resume: bool = False,
) -> str:
    """Run async fairness evaluation."""
    # Get configurations
    aspect_config = ASPECT_DEFINITIONS["fairness"]
    evidence_modality = DATASET_EVIDENCE_MODALITY.get(dataset, "medical image")
    
    # Get data handler
    image_root_path = Path(image_root) if image_root else None
    data_handler = get_data_handler(dataset, image_root_path)
    
    # Load data files
    print(f"Loading baseline results from {baseline_file}...")
    baseline_data = load_jsonl(baseline_file)
    
    print(f"Loading counterfactual results from {cf_file}...")
    cf_data = load_jsonl(cf_file)
    
    # Build pairs by demographic type
    pairs_by_demo = build_fairness_pairs(
        baseline_data, cf_data, data_handler, demographic_type
    )
    
    # Report stats
    for demo_type, pairs in pairs_by_demo.items():
        print(f"  {demo_type}: {len(pairs)} pairs")
    
    # Flatten pairs if evaluating all demographics
    if demographic_type:
        all_pairs = pairs_by_demo[demographic_type]
    else:
        all_pairs = []
        for demo_type in DEMOGRAPHIC_TYPES:
            all_pairs.extend(pairs_by_demo[demo_type])
    
    # Apply filtering
    if end_index is None:
        end_index = len(all_pairs)
    pairs_to_process = all_pairs[start_index:end_index]
    if limit:
        pairs_to_process = pairs_to_process[:limit]
    
    print(f"Total pairs available: {len(pairs_to_process)}")
    
    # Setup output file first (needed for resume)
    if output_file is None:
        output_path = generate_output_path(baseline_file, judge_model, demographic_type)
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load completed entry IDs if resuming
    completed_ids = set()
    if resume:
        completed_ids = load_completed_entry_ids(output_path)
        print(f"Resume mode: Found {len(completed_ids)} already completed entries")
    
    # Build task list
    tasks = []
    skipped_count = 0
    for baseline_item, cf_item in pairs_to_process:
        entry_id = data_handler.get_entry_id(baseline_item)
        demo_type = extract_demographic_type(baseline_item.get("condition", ""))
        
        # Create unique task ID combining question_id and demographic
        task_entry_id = f"{entry_id}_{demo_type}"
        
        # Skip if already completed (resume mode)
        if task_entry_id in completed_ids:
            skipped_count += 1
            continue
        
        tasks.append(EvalTask(
            entry_id=task_entry_id,
            image_path=data_handler.get_full_image_path(baseline_item),
            image_path_relative=data_handler.get_image_path(baseline_item),
            original_q=data_handler.get_question(baseline_item),
            original_a=data_handler.get_response(baseline_item),
            cf_q=data_handler.get_question(cf_item),
            cf_a=data_handler.get_response(cf_item),
            model_name=data_handler.get_model_name(baseline_item),
        ))
    
    if resume:
        print(f"Skipped {skipped_count} already completed entries")
    print(f"Processing {len(tasks)} remaining entries...")
    print(f"Output will be saved to: {output_path}")
    
    if dry_run:
        print("\n[DRY RUN] Would process:")
        for task in tasks[:5]:
            print(f"  Entry {task.entry_id}: {task.original_q[:60]}...")
        if len(tasks) > 5:
            print(f"  ... and {len(tasks) - 5} more entries")
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
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Judge - Fairness Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--baseline_file", "-b",
        required=True,
        help="Path to baseline JSONL file"
    )
    parser.add_argument(
        "--cf_file", "-c",
        required=True,
        help="Path to counterfactual JSONL file"
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        choices=["iu_xray", "ham10000", "harvard_fairvlmed"],
        help="Dataset type for data handling"
    )
    parser.add_argument(
        "--judge_model", "-m",
        default="gemini-2.5-flash",
        help="Judge model to use (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--output_file", "-o",
        default=None,
        help="Output file path (optional, auto-generated if not specified)"
    )
    parser.add_argument(
        "--image_root", "-i",
        default=None,
        help="Root directory for images"
    )
    parser.add_argument(
        "--demographic_type", "-t",
        choices=["gender", "age", "race"],
        default=None,
        help="Filter to specific demographic type (default: evaluate all)"
    )
    parser.add_argument(
        "--start_index", "-s",
        type=int,
        default=0,
        help="Start index for processing (default: 0)"
    )
    parser.add_argument(
        "--end_index", "-e",
        type=int,
        default=None,
        help="End index for processing (default: all)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of entries to process"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent API calls (default: 10)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview what would be processed without running"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from existing output file, skipping already completed entries"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 60)
    print("LLM JUDGE - FAIRNESS EVALUATION")
    print("=" * 60)
    print(f"Baseline file: {args.baseline_file}")
    print(f"CF file: {args.cf_file}")
    print(f"Dataset: {args.dataset}")
    print(f"Evidence modality: {DATASET_EVIDENCE_MODALITY.get(args.dataset, 'medical image')}")
    print(f"Judge model: {args.judge_model}")
    print(f"Demographic filter: {args.demographic_type or 'all'}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Dry run: {args.dry_run}")
    print(f"Resume: {args.resume}")
    print("=" * 60 + "\n")
    
    # Run evaluation
    output_path = asyncio.run(run_async_evaluation(
        baseline_file=args.baseline_file,
        cf_file=args.cf_file,
        dataset=args.dataset,
        judge_model=args.judge_model,
        output_file=args.output_file,
        image_root=args.image_root,
        demographic_type=args.demographic_type,
        start_index=args.start_index,
        end_index=args.end_index,
        limit=args.limit,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        resume=args.resume,
    ))
    
    print(f"\nEvaluation complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
