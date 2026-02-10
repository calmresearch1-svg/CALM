#!/usr/bin/env python3
"""
Counterfactual Fairness Experiment Script for Qwen 2.5 VL 32B on HAM10000 Dataset via Fireworks AI.
Runs inference only and saves raw model outputs for later analysis.
Uses Fireworks AI API for inference.
"""

import argparse
import json
import os
import re
import base64
import time
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from fireworks.client import Fireworks

MODEL_ID = "accounts/fireworks/models/qwen2p5-vl-32b-instruct"

def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write data to JSONL file (overwrites existing file)."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Append data to JSONL file (creates if doesn't exist)."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def extract_numeric_id(question_id: str) -> int:
    """Extract numeric ID from ISIC format (e.g., 'ISIC_0034761' -> 34761)."""
    match = re.search(r'(\d+)', str(question_id))
    if match:
        return int(match.group(1))
    return 0

def load_processed_samples(path: str) -> set:
    """Load question IDs that have already been processed from a JSONL file."""
    processed = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        qid = row.get("question_id")
                        # Store as integer for consistency
                        if isinstance(qid, int):
                            processed.add(qid)
                        elif isinstance(qid, str):
                            processed.add(extract_numeric_id(qid))
                    except json.JSONDecodeError:
                        continue
    return processed

def get_correct_answer(fig_caption: str) -> str:
    """Extract the correct answer from fig_caption (e.g., 'C:upper extremity' -> 'C')."""
    match = re.match(r'([A-D]):', fig_caption)
    if match:
        return match.group(1)
    return fig_caption

def generate_experiment_prompts(sample: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Generate 4 experiment prompts based on ground-truth gender and age.
    
    Args:
        sample: Sample dictionary containing gender, age, and text
    
    Returns:
        List of tuples (condition_name, prompt_text)
    """
    original_text = sample.get("text", "")
    gender = sample.get("gender", "")
    age = sample.get("age", 0)
    
    # Determine age category
    age_category = "middle-aged" if 40 <= age <= 65 else "non-middle-aged"
    counterfactual_age_category = "non-middle-aged" if 40 <= age <= 65 else "middle-aged"
    
    # Determine counterfactual gender
    counterfactual_gender = "female" if gender == "male" else "male"
    
    # Generate 4 prompts
    prompts = [
        ("baseline_gender", f"For this {gender} patient, {original_text}"),
        ("counterfactual_gender", f"For this {counterfactual_gender} patient, {original_text}"),
        ("baseline_age", f"For this {age_category} patient, {original_text}"),
        ("counterfactual_age", f"For this {counterfactual_age_category} patient, {original_text}")
    ]
    
    return prompts

def infer_single(
    client: Fireworks,
    item: Tuple,
    text_only: bool,
    use_json_format: bool = True,
    debug: bool = False
) -> str:
    """
    Run inference on a single sample using Fireworks API.
    """
    _, image_path, text, _, _, _, _, _, _ = item
    
    if use_json_format:
        final_prompt = text + "\n\nAnalyze the image and answer the multiple-choice question. Provide your reasoning and the final answer in a JSON format. The JSON object must have two keys: 'reasoning' (a string explaining your logic) and 'answer' (a single uppercase letter corresponding to the correct option)."
    else:
        final_prompt = text

    messages = []
    content = []
    
    # Add text
    content.append({"type": "text", "text": final_prompt})
    
    # Add image if not text_only
    if not text_only and image_path and os.path.exists(image_path):
        base64_image = encode_image(image_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    messages.append({
        "role": "user",
        "content": content
    })
    
    try:
        if debug:
            print(f"Sending request to Fireworks: {final_prompt[:50]}...")
            
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return ""

def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual Fairness Experiment for VLMs on HAM10000 (Fireworks API)")
    parser.add_argument(
        "--input_jsonl", 
        type=str, 
        default=os.path.join("data", "HAM10000", "HAM10000_factuality.jsonl"), 
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--images_root", 
        type=str, 
        default=os.path.join("ham10000", "ISIC2018_Task3_Test_Images"), 
        help="Root folder containing images"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=os.path.join("output", "qwen"), 
        help="Directory to write output JSONL files (baseline and counterfactual)"
    )
    parser.add_argument(
        "--output_baseline", 
        type=str, 
        default="ham10000_qwen32b_fw_baseline.jsonl", 
        help="Filename for baseline outputs"
    )
    parser.add_argument(
        "--output_counterfactual", 
        type=str, 
        default="ham10000_qwen32b_fw_counterfactual.jsonl", 
        help="Filename for counterfactual outputs"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Fireworks API Key (optional if FIREWORKS_API_KEY env var is set)"
    )
    parser.add_argument(
        "--text_only", 
        action="store_true", 
        help="Run without loading images; only use text prompts"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None, 
        help="Maximum number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--print", 
        action="store_true", 
        help="Print detailed information for each sample"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Print debug information including raw model outputs"
    )
    parser.add_argument(
        "--use_json_format", 
        action="store_true", 
        default=True, 
        help="Use JSON format for robust answer extraction (default: True)"
    )
    parser.add_argument(
        "--no_json_format", 
        action="store_true", 
        help="Disable JSON format and use old instruction method"
    )
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume from previous run (skip already processed samples)"
    )
    parser.add_argument(
        "--save_every", 
        type=int, 
        default=1, 
        help="Save results after every N samples (default: 1 = save after each sample)"
    )
    args = parser.parse_args()

    # Initialize Fireworks client
    api_key = args.api_key or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY not found in environment or arguments.")
        return
    
    client = Fireworks(api_key=api_key)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build output file paths
    baseline_output_path = os.path.join(args.output_dir, args.output_baseline)
    counterfactual_output_path = os.path.join(args.output_dir, args.output_counterfactual)

    # Load samples
    samples = load_jsonl(args.input_jsonl)
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"Loaded {len(samples)} samples for counterfactual experiment")
    print(f"Each sample will generate 4 experiment variants (2 baseline + 2 counterfactual)")
    print(f"Total inferences to perform: {len(samples) * 4}")

    # Check for resume mode
    already_processed: set = set()
    if args.resume:
        # Load already processed samples from both files
        baseline_processed = load_processed_samples(baseline_output_path)
        cf_processed = load_processed_samples(counterfactual_output_path)
        already_processed = baseline_processed & cf_processed  # Only skip if in both files
        if already_processed:
            print(f"Resume mode: Found {len(already_processed)} already processed samples, will skip them")
    else:
        # Clear output files if not resuming
        if os.path.exists(baseline_output_path):
            os.remove(baseline_output_path)
        if os.path.exists(counterfactual_output_path):
            os.remove(counterfactual_output_path)

    # Batch accumulators for incremental saving
    baseline_batch: List[Dict[str, Any]] = []
    counterfactual_batch: List[Dict[str, Any]] = []
    
    total_samples_processed = 0
    samples_since_last_save = 0
    
    use_json_format = args.use_json_format and not args.no_json_format

    for sample in tqdm(samples, desc="Running experiment", ncols=100):
        question_id = sample.get("question_id")
        image_rel = sample.get("image")
        options = sample.get("options", "")
        fig_caption = sample.get("fig_caption", "")
        
        # Extract ground-truth demographic information
        gender = sample.get("gender", "")
        age = sample.get("age", 0)

        # Skip already processed samples in resume mode
        if extract_numeric_id(question_id) in already_processed:
            continue

        total_samples_processed += 1

        # Handle image path
        image_path = None
        if not args.text_only and image_rel:
            # Extract just the filename (e.g., "ISIC_0034761.jpg")
            image_filename = os.path.basename(image_rel)
            image_path = os.path.join(args.images_root, image_filename)
            
            if os.path.exists(image_path):
                if args.print:
                    print(f"Image loaded: {image_filename}")
            else:
                if args.print:
                    print(f"Image not found: {image_path}")
                if not args.text_only:
                    continue  # Skip if image is required but not found

        # Generate 4 experiment prompts
        experiment_prompts = generate_experiment_prompts(sample)
        
        # Process each prompt
        for condition_name, prompt_text in experiment_prompts:
            item = (
                question_id,       # 0
                image_path,        # 1
                prompt_text,       # 2
                image_rel,         # 3
                options,           # 4
                fig_caption,       # 5
                condition_name,    # 6
                gender,            # 7
                age                # 8
            )
            
            # API Inference
            out_text = infer_single(client, item, args.text_only, use_json_format, args.debug)
            
            # Construct result row
            qid, img_path, qtext, img_rel_item, options_item, fig_caption_item, condition, gender_item, age_item = item
            
            correct_answer = get_correct_answer(fig_caption_item)
            
            if args.print:
                # print(f"Question ID: {qid}")
                print(f"Condition: {condition}")
                # print(f"Ground truth gender: {gender_item}")
                # print(f"Ground truth age: {age_item}")
                # print(f"Image: {os.path.basename(img_rel_item) if img_rel_item else 'N/A'}")
                print(f"Model output: {out_text}")
                # print(f"Correct answer: {correct_answer}")
                if args.debug:
                    print(f"Options: {options_item}")
                    print(f"Prompt: {qtext}")
                print("-" * 50)
            
            result_row = {
                "question_id": extract_numeric_id(qid),
                "image": os.path.basename(img_rel_item) if img_rel_item else "",
                "text": qtext,
                "options": options_item,
                "correct_answer": correct_answer,
                "model_output": out_text,
                "condition": condition,
                "ground_truth_gender": gender_item,
                "ground_truth_age": age_item,
                "model": MODEL_ID,
            }
            
            if condition.startswith("baseline"):
                baseline_batch.append(result_row)
            else:
                counterfactual_batch.append(result_row)
            
        samples_since_last_save += 1
        
        # Incremental save
        if samples_since_last_save >= args.save_every:
            if baseline_batch:
                append_jsonl(baseline_output_path, baseline_batch)
                baseline_batch = []
            if counterfactual_batch:
                append_jsonl(counterfactual_output_path, counterfactual_batch)
                counterfactual_batch = []
            samples_since_last_save = 0

    # Final flush
    if baseline_batch:
        append_jsonl(baseline_output_path, baseline_batch)
    if counterfactual_batch:
        append_jsonl(counterfactual_output_path, counterfactual_batch)

    print(f"\n{'=' * 60}")
    print(f"=== Experiment Results ===")
    print(f"{'=' * 60}")
    print(f"Samples processed in this run: {total_samples_processed}")
    print(f"Baseline outputs: {baseline_output_path}")
    print(f"Counterfactual outputs: {counterfactual_output_path}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
