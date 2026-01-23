#!/usr/bin/env python3
"""
LLM Judge - Privacy Evaluation Script

Evaluates privacy by assessing individual model responses (no image, no paired comparison).
Each response is evaluated independently for privacy compliance.

Usage:
    python llj_privacy.py \
        --input_file JSONs/Privacy/Base/qwen_evaluation_results.json \
        --judge_model gemini-2.5-flash
"""

import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Any

from dotenv import load_dotenv

load_dotenv()

from llm_judge import (
    ASPECT_DEFINITIONS,
    AsyncLLMJudge,
    SingleEvalTask,
)

# Fixed outputs directory
OUTPUTS_ROOT = Path(__file__).parent / "LLJ_Outputs"


def load_json_data(file_path: str) -> List[Dict]:
    """Load data from JSON file (array format)."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_completed_entry_ids(output_file: Path) -> set:
    """
    Load entry IDs that have already been processed from an existing output file.
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


def generate_output_path(input_file: str, judge_model: str) -> Path:
    """Generate output file path."""
    base_name = Path(input_file).stem
    model_clean = judge_model.replace("/", "-").replace(".", "-")
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{base_name}_LLJ_result_{model_clean}_{date_str}.jsonl"
    
    output_dir = OUTPUTS_ROOT / "Privacy"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir / filename


def get_entry_id(item: Dict) -> str:
    """Generate unique entry ID from subject_id, hadm_id, and question_id."""
    subject_id = item.get("subject_id", "")
    hadm_id = item.get("hadm_id", "")
    question_id = item.get("question_id", "")
    return f"{subject_id}_{hadm_id}_{question_id}"


async def run_async_evaluation(
    input_file: str,
    judge_model: str = "gemini-2.5-flash",
    output_file: str = None,
    start_index: int = 0,
    end_index: int = None,
    limit: int = None,
    concurrency: int = 5,
    dry_run: bool = False,
    resume: bool = False,
) -> str:
    """Run async privacy evaluation using the shared AsyncLLMJudge."""
    # Get configurations
    aspect_config = ASPECT_DEFINITIONS["privacy"]
    
    # Load data file
    print(f"Loading data from {input_file}...")
    data = load_json_data(input_file)
    print(f"Loaded {len(data)} entries")
    
    # Apply filtering
    if end_index is None:
        end_index = len(data)
    data_to_process = data[start_index:end_index]
    if limit:
        data_to_process = data_to_process[:limit]
    
    print(f"Total entries to evaluate: {len(data_to_process)}")
    
    # Setup output file first (needed for resume)
    if output_file is None:
        output_path = generate_output_path(input_file, judge_model)
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load completed entry IDs if resuming
    completed_ids = set()
    if resume:
        completed_ids = load_completed_entry_ids(output_path)
        print(f"Resume mode: Found {len(completed_ids)} already completed entries")
    
    # Build task list using SingleEvalTask from base_judge
    tasks = []
    skipped_count = 0
    for item in data_to_process:
        entry_id = get_entry_id(item)
        
        # Skip if already completed (resume mode)
        if entry_id in completed_ids:
            skipped_count += 1
            continue
        
        tasks.append(SingleEvalTask(
            entry_id=entry_id,
            image_path=None,  # No image for privacy evaluation
            image_path_relative=None,
            question=item.get("question", ""),
            response=item.get("response", ""),
            model_name=None,  # Could extract from filename if needed
        ))
    
    if resume:
        print(f"Skipped {skipped_count} already completed entries")
    print(f"Processing {len(tasks)} remaining entries...")
    print(f"Output will be saved to: {output_path}")
    
    if dry_run:
        print("\n[DRY RUN] Would process:")
        for task in tasks[:5]:
            print(f"  Entry {task.entry_id}: {task.question[:60]}...")
        if len(tasks) > 5:
            print(f"  ... and {len(tasks) - 5} more entries")
        return str(output_path)
    
    if len(tasks) == 0:
        print("No new entries to process!")
        return str(output_path)
    
    # Initialize async judge using the shared infrastructure
    start_time = time.time()
    
    judge = AsyncLLMJudge(
        aspect_config=aspect_config,
        judge_model=judge_model,
        evidence_modality="text",  # No image, text-only evaluation
        concurrency=concurrency,
    )
    
    # Use evaluate_single_batch for individual evaluations
    await judge.evaluate_single_batch(tasks, output_path)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/len(tasks):.2f}s per entry)")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Judge - Privacy Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input_file", "-i",
        required=True,
        help="Path to input JSON file (array of objects)"
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
        help="Number of concurrent API calls (default: 5)"
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
    print("LLM JUDGE - PRIVACY EVALUATION")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Judge model: {args.judge_model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Dry run: {args.dry_run}")
    print(f"Resume: {args.resume}")
    print("=" * 60 + "\n")
    
    # Run evaluation
    output_path = asyncio.run(run_async_evaluation(
        input_file=args.input_file,
        judge_model=args.judge_model,
        output_file=args.output_file,
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
