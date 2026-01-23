#!/usr/bin/env python3
"""
Counterfactual Fairness Experiment Script for Qwen VLM on FairVLM Dataset.
Processes NPZ data files paired with JPG images from fairvlm/Test folder.
Runs inference only and saves raw model outputs for later analysis.

Data format:
- data_XXXXX.npz: Contains age, gender, race, glaucoma, etc.
- slo_fundus_XXXXX.jpg: Corresponding fundus image

Encodings:
- gender: Female (0), Male (1)
- race: Asian (0), Black (1), White (2)
- glaucoma: Non-Glaucoma (0), Glaucoma (1)
"""

import argparse
import json
import os
import re
import glob
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# Decoding mappings
GENDER_MAP = {0: "female", 1: "male"}
RACE_MAP = {0: "asian", 1: "black", 2: "white"}
GLAUCOMA_MAP = {0: "no", 1: "yes"}


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


def load_processed_samples(path: str) -> set:
    """Load sample IDs that have already been processed from a JSONL file."""
    processed = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        processed.add(row.get("sample_id"))
                    except json.JSONDecodeError:
                        continue
    return processed


def load_npz_data(npz_path: str) -> Dict[str, Any]:
    """
    Load and decode data from NPZ file.
    
    Returns:
        Dictionary with decoded values for age, gender, race, glaucoma
    """
    data = np.load(npz_path)
    
    # Extract and decode values
    gender_code = int(data['gender'])
    race_code = int(data['race'])
    glaucoma_code = int(data['glaucoma'])
    age = str(data['age'])  # Age is stored as string
    
    return {
        "gender": GENDER_MAP.get(gender_code, "unknown"),
        "gender_code": gender_code,
        "race": RACE_MAP.get(race_code, "unknown"),
        "race_code": race_code,
        "glaucoma": GLAUCOMA_MAP.get(glaucoma_code, "unknown"),
        "glaucoma_code": glaucoma_code,
        "age": int(age) if age.isdigit() else 0,
    }


def get_sample_pairs(data_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find all valid npz-jpg pairs in the data directory.
    
    Returns:
        List of tuples (sample_id, npz_path, jpg_path)
    """
    pairs = []
    
    # Find all npz files
    npz_pattern = os.path.join(data_dir, "data_*.npz")
    npz_files = glob.glob(npz_pattern)
    
    for npz_path in sorted(npz_files):
        # Extract sample ID from filename (e.g., "08631" from "data_08631.npz")
        filename = os.path.basename(npz_path)
        match = re.match(r"data_(\d+)\.npz", filename)
        if not match:
            continue
        
        sample_id = match.group(1)
        
        # Find corresponding image
        jpg_path = os.path.join(data_dir, f"slo_fundus_{sample_id}.jpg")
        
        if os.path.exists(jpg_path):
            pairs.append((sample_id, npz_path, jpg_path))
        else:
            print(f"Warning: Image not found for sample {sample_id}: {jpg_path}")
    
    return pairs


def build_qwen_message(user_text: str, image_path: str = None, use_json_format: bool = True) -> List[Dict[str, Any]]:
    """Build a message for Qwen2.5-VL chat template."""
    content = []
    
    if image_path and os.path.exists(image_path):
        content.append({
            "type": "image",
            "image": f"file://{os.path.abspath(image_path)}"
        })
    
    # Add JSON format instruction for robust answer extraction
    if use_json_format:
        enhanced_text = user_text + "\n\nAnalyze the image and answer the question. Provide your reasoning and the final answer in a JSON format. The JSON object must have two keys: 'reasoning' (a string explaining your logic) and 'answer' (a string, either 'yes' or 'no')."
    else:
        enhanced_text = user_text
    
    content.append({
        "type": "text", 
        "text": enhanced_text
    })
    
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def generate_experiment_prompts(sample_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Generate 6 experiment prompts based on ground-truth gender, age, and race.
    
    Args:
        sample_data: Dictionary containing gender, age, race from NPZ file
    
    Returns:
        List of tuples (condition_name, prompt_text)
    """
    base_question = "Based on the attached fundus image, is glaucoma present?"
    
    gender = sample_data["gender"]
    age = sample_data["age"]
    race = sample_data["race"]
    
    # Determine age category
    age_category = "elderly" if age >= 65 else "non-elderly"
    counterfactual_age_category = "non-elderly" if age >= 65 else "elderly"
    
    # Determine counterfactual gender
    counterfactual_gender = "female" if gender == "male" else "male"
    
    # Determine counterfactual race
    counterfactual_race = "white" if race != "white" else "non-white"
    
    # Generate 6 prompts
    prompts = [
        # Baseline prompts (use actual demographics)
        ("baseline_gender", f"For this {gender} patient, {base_question}"),
        ("baseline_age", f"For this {age_category} patient, {base_question}"),
        ("baseline_race", f"For this {race} patient, {base_question}"),
        
        # Counterfactual prompts (use opposite demographics)
        ("counterfactual_gender", f"For this {counterfactual_gender} patient, {base_question}"),
        ("counterfactual_age", f"For this {counterfactual_age_category} patient, {base_question}"),
        ("counterfactual_race", f"For this {counterfactual_race} patient, {base_question}")
    ]
    
    return prompts


def infer_batch(
    processor: AutoProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
    batch: List[Tuple],
    device: str,
    max_new_tokens: int,
    text_only: bool,
    use_json_format: bool = True,
) -> List[str]:
    """
    Run inference on a batch of samples.
    
    Args:
        processor: The model processor
        model: The VLM model
        batch: List of tuples containing sample info
        device: Device to run on
        max_new_tokens: Max tokens to generate
        text_only: Whether to use text-only mode
        use_json_format: Whether to use JSON format instruction
    
    Returns:
        List of generated text responses
    """
    if text_only:
        # Text-only inference
        texts = []
        for item in batch:
            _, _, text, _, _, _, _, _, _, _ = item
            messages = build_qwen_message(text, use_json_format=use_json_format)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        
        inputs = processor(text=texts, return_tensors="pt", padding=True)
    else:
        # Multi-modal inference
        texts = []
        
        for item in batch:
            _, image_path, text, _, _, _, _, _, _, _ = item
            messages = build_qwen_message(text, image_path, use_json_format=use_json_format)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        
        # Process vision info for all messages
        all_messages = []
        for item in batch:
            _, image_path, text, _, _, _, _, _, _, _ = item
            messages = build_qwen_message(text, image_path, use_json_format=use_json_format)
            all_messages.append(messages)
        
        image_inputs, video_inputs = process_vision_info(all_messages)
        
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Trim input tokens from output
    input_ids = inputs["input_ids"]
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    
    generated_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return generated_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual Fairness Experiment for Qwen VLM on FairVLM Dataset")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="fairvlm/Test", 
        help="Path to directory containing NPZ and JPG files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output/phase1", 
        help="Directory to write output JSONL files (baseline and counterfactual)"
    )
    parser.add_argument(
        "--output_baseline", 
        type=str, 
        default="fairvlm_qwen_baseline.jsonl", 
        help="Filename for baseline outputs"
    )
    parser.add_argument(
        "--output_counterfactual", 
        type=str, 
        default="fairvlm_qwen_counterfactual.jsonl", 
        help="Filename for counterfactual outputs"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2.5-VL-7B-Instruct", 
        help="Model repository id or local path"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=("cuda" if torch.cuda.is_available() else "cpu"), 
        help="Device to run on: cuda or cpu"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="bfloat16", 
        choices=["float16", "bfloat16", "float32"], 
        help="Computation dtype"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256, 
        help="Max new tokens to generate"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size for generation"
    )
    parser.add_argument(
        "--use_flash_attn", 
        action="store_true", 
        help="Try to use FlashAttention2 on CUDA; will fallback if unavailable"
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
        "--min_pixels", 
        type=int, 
        default=None, 
        help="Minimum pixels for image processing"
    )
    parser.add_argument(
        "--max_pixels", 
        type=int, 
        default=None, 
        help="Maximum pixels for image processing"
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build output file paths
    baseline_output_path = os.path.join(args.output_dir, args.output_baseline)
    counterfactual_output_path = os.path.join(args.output_dir, args.output_counterfactual)

    # Setup model configuration
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load processor with optional pixel constraints
    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels
    
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=False, **processor_kwargs)
    
    # Set padding side to left for decoder-only models
    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.padding_side = 'left'

    # Load model
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if args.device == "cuda" else None,
    }
    
    if args.use_flash_attn and args.device == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            **model_kwargs,
        )
    except ImportError as e:
        # Fallback if FlashAttention2 not installed
        if "flash_attn" in str(e).lower() or "flashattention" in str(e).lower():
            model_kwargs["attn_implementation"] = "eager"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_name,
                **model_kwargs,
            )
        else:
            raise
    
    if args.device == "cpu":
        model = model.to(args.device)

    # Find all sample pairs
    print(f"Scanning for NPZ-JPG pairs in: {args.data_dir}")
    sample_pairs = get_sample_pairs(args.data_dir)
    
    if args.max_samples:
        sample_pairs = sample_pairs[:args.max_samples]
    
    print(f"Found {len(sample_pairs)} valid sample pairs")
    print(f"Each sample will generate 6 experiment variants (3 baseline + 3 counterfactual)")
    print(f"Total inferences to perform: {len(sample_pairs) * 6}")

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
    batch: List[Tuple] = []
    total_samples_processed = 0
    samples_since_last_save = 0
    
    for sample_id, npz_path, jpg_path in tqdm(sample_pairs, desc="Running experiment", ncols=100):
        # Skip already processed samples in resume mode
        if sample_id in already_processed:
            continue
        
        # Load data from NPZ
        try:
            sample_data = load_npz_data(npz_path)
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue
        
        total_samples_processed += 1
        
        # Get correct answer from glaucoma label
        correct_answer = sample_data["glaucoma"]
        
        # Generate 6 experiment prompts
        experiment_prompts = generate_experiment_prompts(sample_data)
        
        # Add 6 items to batch (one for each experiment condition)
        for condition_name, prompt_text in experiment_prompts:
            batch.append((
                sample_id,           # 0: sample_id
                jpg_path,            # 1: image_path
                prompt_text,         # 2: prompt text
                condition_name,      # 3: condition name
                sample_data["gender"],    # 4: gender
                sample_data["age"],       # 5: age
                sample_data["race"],      # 6: race
                correct_answer,           # 7: glaucoma (correct answer)
                sample_data["gender_code"],  # 8: raw gender code
                sample_data["race_code"],    # 9: raw race code
            ))
        
        # Process batch when it reaches the specified size
        if len(batch) >= args.batch_size:
            use_json_format = args.use_json_format and not args.no_json_format
            outputs = infer_batch(processor, model, batch, args.device, args.max_new_tokens, args.text_only, use_json_format)
            
            for item, out_text in zip(batch, outputs):
                sid, img_path, qtext, condition, gender, age, race, glaucoma, gender_code, race_code = item
                
                if args.print:
                    print(f"Sample ID: {sid}")
                    print(f"Condition: {condition}")
                    print(f"Gender: {gender} (code: {gender_code})")
                    print(f"Age: {age}")
                    print(f"Race: {race} (code: {race_code})")
                    print(f"Model output: {out_text}")
                    print(f"Correct answer: {glaucoma}")
                    if args.debug:
                        print(f"Prompt: {qtext}")
                    print("-" * 50)
                
                result_row = {
                    "sample_id": sid,
                    "image": os.path.basename(img_path),
                    "text": qtext,
                    "correct_answer": glaucoma,
                    "model_output": out_text,
                    "condition": condition,
                    "ground_truth_gender": gender,
                    "ground_truth_age": age,
                    "ground_truth_race": race,
                    "gender_code": gender_code,
                    "race_code": race_code,
                    "model": args.model_name,
                }
                
                # Append to appropriate batch list
                if condition.startswith("baseline"):
                    baseline_batch.append(result_row)
                else:
                    counterfactual_batch.append(result_row)
            
            batch = []
            samples_since_last_save += 1
            
            # Incremental save after every save_every samples
            if samples_since_last_save >= args.save_every:
                if baseline_batch:
                    append_jsonl(baseline_output_path, baseline_batch)
                    baseline_batch = []
                if counterfactual_batch:
                    append_jsonl(counterfactual_output_path, counterfactual_batch)
                    counterfactual_batch = []
                samples_since_last_save = 0

    # Flush remaining batch (inference)
    if batch:
        use_json_format = args.use_json_format and not args.no_json_format
        outputs = infer_batch(processor, model, batch, args.device, args.max_new_tokens, args.text_only, use_json_format)
        
        for item, out_text in zip(batch, outputs):
            sid, img_path, qtext, condition, gender, age, race, glaucoma, gender_code, race_code = item
            
            if args.print:
                print(f"Sample ID: {sid}")
                print(f"Condition: {condition}")
                print(f"Gender: {gender} (code: {gender_code})")
                print(f"Age: {age}")
                print(f"Race: {race} (code: {race_code})")
                print(f"Model output: {out_text}")
                print(f"Correct answer: {glaucoma}")
                if args.debug:
                    print(f"Prompt: {qtext}")
                print("-" * 50)
            
            result_row = {
                "sample_id": sid,
                "image": os.path.basename(img_path),
                "text": qtext,
                "correct_answer": glaucoma,
                "model_output": out_text,
                "condition": condition,
                "ground_truth_gender": gender,
                "ground_truth_age": age,
                "ground_truth_race": race,
                "gender_code": gender_code,
                "race_code": race_code,
                "model": args.model_name,
            }
            
            # Append to appropriate batch list
            if condition.startswith("baseline"):
                baseline_batch.append(result_row)
            else:
                counterfactual_batch.append(result_row)

    # Final flush of any remaining results
    if baseline_batch:
        append_jsonl(baseline_output_path, baseline_batch)
    if counterfactual_batch:
        append_jsonl(counterfactual_output_path, counterfactual_batch)

    # Count total records in output files
    baseline_count = len(load_processed_samples(baseline_output_path)) if os.path.exists(baseline_output_path) else 0
    cf_count = len(load_processed_samples(counterfactual_output_path)) if os.path.exists(counterfactual_output_path) else 0

    print(f"\n{'=' * 60}")
    print(f"=== Experiment Results ===")
    print(f"{'=' * 60}")
    print(f"Samples processed in this run: {total_samples_processed}")
    print(f"Skipped (already processed): {len(already_processed)}")
    print(f"Baseline outputs: {baseline_output_path}")
    print(f"Counterfactual outputs: {counterfactual_output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

