#!/usr/bin/env python3
"""
LLM Judge - Trustfulness/Factuality Evaluation Script

Evaluates trustfulness or factuality aspects by assessing individual model responses
with associated images.

The script handles the specific JSON structure from trustfulness datasets:
- Each line contains both original and modified questions with responses:
  - pure_prompt: original question
  - pure_prediction: model's response to original question  
  - counterfactual_prompt: modified question
  - cf_prediction: model's response to modified question
- Image field can be 'image' (single path) or 'images' (list of paths)

For each input line, two evaluations are performed:
1. pure_prompt + pure_prediction
2. counterfactual_prompt + cf_prediction

Usage:
    python llj_trustfulness.py \
        --input_file JSONs/Trustfulness/data.jsonl \
        --aspect trustfulness \
        --judge_model gemini-2.5-flash
        
    python llj_trustfulness.py \
        --input_file JSONs/Trustfulness/data.jsonl \
        --aspect factuality \
        --judge_model gemini-2.5-flash
"""

import asyncio
import argparse
import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

from llm_judge import (
    ASPECT_DEFINITIONS,
    DATASET_EVIDENCE_MODALITY,
)
from llm_judge.config import OUTPUTS_ROOT, DEFAULT_OUTPUT_FORMAT
from llm_judge.models import ModelFactory
from llm_judge.prompts import PromptBuilder
from llm_judge.parsers import extract_score
from llm_judge.utils import encode_image_base64, get_image_mime_type


# Trustfulness aspect mapping
TRUSTFULNESS_ASPECTS = {
    "trustfulness": "trustfulness",
    "factuality": "factuality",
}


@dataclass
class TrustfulnessEvalTask:
    """Evaluation task for trustfulness/factuality analysis."""
    entry_id: str
    task_type: str  # "pure" or "counterfactual"
    image_paths: List[str]  # List of image paths (can be single or multiple)
    question: str
    response: str
    model_name: Optional[str] = None


def load_trustfulness_data(file_path: str) -> List[Dict]:
    """
    Load trustfulness data from JSONL file.
    
    Returns:
        List of data entries
    """
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
    return entries


def get_image_paths(item: Dict, image_root: Optional[str] = None) -> List[str]:
    """
    Extract image paths from an item.
    Handles both 'image' (single path) and 'images' (list of paths) fields.
    
    Args:
        item: Data item dictionary
        image_root: Optional root directory to prepend to relative paths
    
    Returns:
        List of full image paths
    """
    paths = []
    
    # Check for 'images' field first (list of paths)
    if "images" in item:
        images = item["images"]
        if isinstance(images, list):
            paths = images
        elif isinstance(images, str):
            paths = [images]
    # Check for 'image' field (single path)  
    elif "image" in item:
        image = item["image"]
        if isinstance(image, str):
            paths = [image]
        elif isinstance(image, list):
            paths = image
    
    # Prepend image_root if provided
    if image_root and paths:
        from os.path import join, isabs
        paths = [p if isabs(p) else join(image_root, p) for p in paths]
    
    return paths


def load_completed_entry_ids(output_file: Path) -> set:
    """
    Load entry IDs that have already been processed from an existing output file.
    Entry IDs include the task type suffix (e.g., "123_pure", "123_counterfactual").
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


def generate_output_path(input_file: str, aspect: str, judge_model: str) -> Path:
    """Generate output file path."""
    base_name = Path(input_file).stem
    model_clean = judge_model.replace("/", "-").replace(".", "-")
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{base_name}_{aspect}_LLJ_result_{model_clean}_{date_str}.jsonl"
    
    # Organize by Trustfulness subcategory
    output_dir = OUTPUTS_ROOT / "Trustfulness" / aspect
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir / filename


def build_multimodal_content_multi_image(
    text: str, 
    image_paths: List[str], 
    strict_images: bool = False
) -> List[Dict[str, Any]]:
    """
    Build multimodal content list for LangChain messages with multiple images.
    
    Args:
        text: Text content
        image_paths: List of paths to images
        strict_images: If True, raise error when images not found; otherwise warn
        
    Returns:
        List of content dictionaries compatible with LangChain
    """
    import os
    content = [{"type": "text", "text": text}]
    
    for image_path in image_paths:
        if not image_path:
            continue
            
        if not os.path.exists(image_path):
            if strict_images:
                raise FileNotFoundError(f"Image not found: {image_path}")
            else:
                print(f"Warning: Image not found, skipping: {image_path}")
                continue
        
        mime_type = get_image_mime_type(image_path)
        base64_img = encode_image_base64(image_path)
        
        if base64_img:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}
            })
    
    return content


class AsyncTrustfulnessJudge:
    """
    Async LLM Judge for trustfulness/factuality evaluation.
    
    Handles multiple images per evaluation task.
    """
    
    def __init__(
        self,
        aspect_config,
        judge_model: str = "gemini-2.5-flash",
        evidence_modality: str = "medical image",
        concurrency: int = 10,
        max_retries: int = 8,
        strict_images: bool = False,
    ):
        self.aspect_config = aspect_config
        self.judge_model = judge_model
        self.evidence_modality = evidence_modality
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.strict_images = strict_images
        
        # Initialize model
        self.model = ModelFactory.create(judge_model)
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(
            aspect_instruction=aspect_config.get_instruction(evidence_modality),
            output_format=DEFAULT_OUTPUT_FORMAT
        )
        
        # Semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(concurrency)
    
    async def evaluate_task_with_write(
        self, 
        task: TrustfulnessEvalTask,
        output_file: Path,
        file_lock: asyncio.Lock,
        pbar,
    ) -> Dict[str, Any]:
        """Evaluate a single task and write result immediately."""
        async with self.semaphore:
            try:
                result = await self._call_judge_async(task)
            except Exception as e:
                print(f"\nError on entry {task.entry_id}: {e}")
                result = {"raw_response": f"Error: {str(e)}", "score": None, "prompt": None}
            
            # Build output entry
            output_entry = {
                "entry_id": task.entry_id,
                "task_type": task.task_type,
                "image_paths": task.image_paths,
                "llm_judge_score": result["score"],
                "llm_judge_raw_response": result["raw_response"],
                "prompt": result.get("prompt"),
                "question": task.question,
                "response": task.response,
                "judge_model": self.judge_model,
                "evaluated_model": task.model_name,
                "aspect": self.aspect_config.name,
            }
            
            # Write immediately with lock
            async with file_lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(output_entry) + "\n")
            
            pbar.update(1)
            return output_entry
    
    async def _call_judge_async(self, task: TrustfulnessEvalTask) -> Dict[str, Any]:
        """Make async API call for single Q&A evaluation with multiple images."""
        # Build single prompt
        prompt = self.prompt_builder.build_single_prompt(task.question, task.response)
        
        # Build content with multiple images
        include_images = self.aspect_config.requires_image and task.image_paths
        content = build_multimodal_content_multi_image(
            prompt, 
            task.image_paths if include_images else [],
            strict_images=self.strict_images
        )
        message = HumanMessage(content=content)
        
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(self.model.invoke, [message])
                
                raw_text = response.content if hasattr(response, 'content') else str(response)
                score = extract_score(raw_text)
                
                return {"raw_response": raw_text, "score": score, "prompt": prompt}
                
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate limit" in error_str.lower()
                
                retry_match = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", error_str)
                
                if retry_match:
                    delay = int(retry_match.group(1)) + 1
                elif is_rate_limit:
                    delay = min(2 ** (attempt + 2), 120)
                else:
                    delay = min(2 ** attempt, 30)
                
                print(f"\n  Attempt {attempt + 1}/{self.max_retries} failed for {task.entry_id}: {error_str[:80]}")
                print(f"  {'Rate limit' if is_rate_limit else 'Error'}. Waiting {delay}s before retry...")
                await asyncio.sleep(delay)
        
        return {"raw_response": '{"reasoning": "Failed after all retries", "score": 0}', "score": None, "prompt": prompt}
    
    async def evaluate_batch(
        self,
        tasks: List[TrustfulnessEvalTask],
        output_file: Path,
    ) -> List[Dict[str, Any]]:
        """Evaluate tasks concurrently, writing results as they complete."""
        print(f"Processing {len(tasks)} entries with concurrency={self.concurrency}...")
        print(f"Results will be written incrementally to: {output_file}")
        
        file_lock = asyncio.Lock()
        results = []
        
        with tqdm_asyncio(total=len(tasks), desc="Evaluating") as pbar:
            coroutines = [
                self.evaluate_task_with_write(task, output_file, file_lock, pbar)
                for task in tasks
            ]
            
            for coro in asyncio.as_completed(coroutines):
                try:
                    result = await coro
                    results.append(result)
                except Exception as e:
                    print(f"\nTask failed: {e}")
        
        return results


async def run_async_evaluation(
    input_file: str,
    aspect: str,
    judge_model: str = "gemini-2.5-flash",
    dataset: str = None,
    output_file: str = None,
    image_root: str = None,
    start_index: int = 0,
    end_index: int = None,
    limit: int = None,
    concurrency: int = 5,
    dry_run: bool = False,
    resume: bool = False,
    strict_images: bool = False,
) -> str:
    """Run async trustfulness/factuality evaluation."""
    
    # Validate and get aspect config key
    if aspect not in TRUSTFULNESS_ASPECTS:
        raise ValueError(f"Unknown aspect: {aspect}. Must be one of: {list(TRUSTFULNESS_ASPECTS.keys())}")
    
    aspect_key = TRUSTFULNESS_ASPECTS[aspect]
    aspect_config = ASPECT_DEFINITIONS[aspect_key]
    
    # Get evidence modality if dataset specified
    evidence_modality = DATASET_EVIDENCE_MODALITY.get(dataset, "medical image") if dataset else "medical image"
    
    # Load data file
    print(f"Loading data from {input_file}...")
    data = load_trustfulness_data(input_file)
    print(f"Loaded {len(data)} entries")
    
    # Apply filtering
    if end_index is None:
        end_index = len(data)
    data_to_process = data[start_index:end_index]
    if limit:
        data_to_process = data_to_process[:limit]
    
    print(f"Total entries to process: {len(data_to_process)}")
    print(f"Total evaluations (2 per entry): {len(data_to_process) * 2}")
    
    # Setup output file first (needed for resume)
    if output_file is None:
        output_path = generate_output_path(input_file, aspect, judge_model)
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load completed entry IDs if resuming
    completed_ids = set()
    if resume:
        completed_ids = load_completed_entry_ids(output_path)
        print(f"Resume mode: Found {len(completed_ids)} already completed evaluations")
    
    # Build task list - 2 tasks per data entry (pure and counterfactual)
    tasks = []
    skipped_count = 0
    
    for idx, item in enumerate(data_to_process):
        # Generate base entry ID (use index or sample_id if available)
        base_id = str(item.get("sample_id", item.get("id", idx)))
        
        # Get image paths (handles both 'image' and 'images' fields)
        image_paths = get_image_paths(item, image_root)
        
        # Get model name if available
        model_name = item.get("model_name", item.get("model", None))
        
        # Task 1: Pure (original) question
        pure_entry_id = f"{base_id}_pure"
        if pure_entry_id not in completed_ids:
            pure_question = item.get("pure_prompt", "")
            pure_response = item.get("pure_prediction", "")
            
            if pure_question and pure_response:
                tasks.append(TrustfulnessEvalTask(
                    entry_id=pure_entry_id,
                    task_type="pure",
                    image_paths=image_paths,
                    question=pure_question,
                    response=pure_response,
                    model_name=model_name,
                ))
        else:
            skipped_count += 1
        
        # Task 2: Counterfactual question
        cf_entry_id = f"{base_id}_counterfactual"
        if cf_entry_id not in completed_ids:
            cf_question = item.get("counterfactual_prompt", "")
            cf_response = item.get("cf_prediction", "")
            
            if cf_question and cf_response:
                tasks.append(TrustfulnessEvalTask(
                    entry_id=cf_entry_id,
                    task_type="counterfactual",
                    image_paths=image_paths,
                    question=cf_question,
                    response=cf_response,
                    model_name=model_name,
                ))
        else:
            skipped_count += 1
    
    if resume:
        print(f"Skipped {skipped_count} already completed evaluations")
    print(f"Processing {len(tasks)} remaining evaluations...")
    print(f"Output will be saved to: {output_path}")
    
    if dry_run:
        print("\n[DRY RUN] Would process:")
        for task in tasks[:5]:
            print(f"  Entry {task.entry_id} ({task.task_type}): {task.question[:60]}...")
            print(f"    Images: {task.image_paths[:2]}{'...' if len(task.image_paths) > 2 else ''}")
        if len(tasks) > 5:
            print(f"  ... and {len(tasks) - 5} more evaluations")
        return str(output_path)
    
    if len(tasks) == 0:
        print("No new evaluations to process!")
        return str(output_path)
    
    # Initialize async judge and run
    start_time = time.time()
    
    judge = AsyncTrustfulnessJudge(
        aspect_config=aspect_config,
        judge_model=judge_model,
        evidence_modality=evidence_modality,
        concurrency=concurrency,
        strict_images=strict_images,
    )
    
    await judge.evaluate_batch(tasks, output_path)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/len(tasks):.2f}s per evaluation)")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Judge - Trustfulness/Factuality Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate trustfulness on data
    python llj_trustfulness.py -i JSONs/Trustfulness/data.jsonl -a trustfulness
    
    # Evaluate factuality on data
    python llj_trustfulness.py -i JSONs/Trustfulness/data.jsonl -a factuality
    
    # Dry run to preview processing
    python llj_trustfulness.py -i data.jsonl -a trustfulness --dry_run
    
    # Resume interrupted evaluation
    python llj_trustfulness.py -i data.jsonl -a trustfulness --resume
"""
    )
    
    parser.add_argument(
        "--input_file", "-i",
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--aspect", "-a",
        required=True,
        choices=["trustfulness", "factuality"],
        help="Aspect to evaluate"
    )
    parser.add_argument(
        "--judge_model", "-m",
        default="gemini-2.5-flash",
        help="Judge model to use (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--dataset", "-d",
        default=None,
        choices=["iu_xray", "ham10000", "harvard_fairvlmed"],
        help="Dataset name for evidence modality (optional)"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Root directory to prepend to relative image paths"
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
    parser.add_argument(
        "--strict_images",
        action="store_true",
        help="Raise error if image files are not found (default: warn and continue)"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 60)
    print(f"LLM JUDGE - TRUSTFULNESS EVALUATION ({args.aspect.upper()})")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Aspect: {args.aspect}")
    print(f"Judge model: {args.judge_model}")
    print(f"Dataset: {args.dataset or 'Not specified'}")
    print(f"Image root: {args.image_root or 'Not specified'}")
    print(f"Strict images: {args.strict_images}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Dry run: {args.dry_run}")
    print(f"Resume: {args.resume}")
    print("=" * 60 + "\n")
    
    # Run evaluation
    output_path = asyncio.run(run_async_evaluation(
        input_file=args.input_file,
        aspect=args.aspect,
        judge_model=args.judge_model,
        dataset=args.dataset,
        output_file=args.output_file,
        image_root=args.image_root,
        start_index=args.start_index,
        end_index=args.end_index,
        limit=args.limit,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        resume=args.resume,
        strict_images=args.strict_images,
    ))
    
    print(f"\nEvaluation complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
