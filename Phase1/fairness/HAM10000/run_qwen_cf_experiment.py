#!/usr/bin/env python3
"""
Counterfactual Fairness Experiment Script for Medical VLMs on HAM10000 Dataset.
Runs inference only and saves raw model outputs for later analysis.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


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


def extract_numeric_id(question_id: str) -> int:
    """Extract numeric ID from ISIC format (e.g., 'ISIC_0034761' -> 34761)."""
    match = re.search(r'(\d+)', question_id)
    if match:
        return int(match.group(1))
    return 0


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
        enhanced_text = user_text + "\n\nAnalyze the image and answer the multiple-choice question. Provide your reasoning and the final answer in a JSON format. The JSON object must have two keys: 'reasoning' (a string explaining your logic) and 'answer' (a single uppercase letter corresponding to the correct option)."
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


def load_image(path: str) -> Image.Image:
    """Load and convert image to RGB."""
    return Image.open(path).convert("RGB")


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
            _, _, text, _, _, _, _, _, _ = item
            messages = build_qwen_message(text, use_json_format=use_json_format)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        
        inputs = processor(text=texts, return_tensors="pt", padding=True)
    else:
        # Multi-modal inference
        texts = []
        
        for item in batch:
            _, image_path, text, _, _, _, _, _, _ = item
            messages = build_qwen_message(text, image_path, use_json_format=use_json_format)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        
        # Process vision info for all messages
        all_messages = []
        for item in batch:
            _, image_path, text, _, _, _, _, _, _ = item
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual Fairness Experiment for VLMs on HAM10000")
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
        default="ham10000_qwen_baseline.jsonl", 
        help="Filename for baseline outputs"
    )
    parser.add_argument(
        "--output_counterfactual", 
        type=str, 
        default="ham10000_qwen_counterfactual.jsonl", 
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
    batch: List[Tuple] = []
    total_samples_processed = 0
    samples_since_last_save = 0
    
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
        
        # Add 4 items to batch (one for each experiment condition)
        for condition_name, prompt_text in experiment_prompts:
            batch.append((
                question_id,       # 0: question_id
                image_path,        # 1: image_path
                prompt_text,       # 2: prompt text
                image_rel,         # 3: original image reference from JSONL
                options,           # 4: options
                fig_caption,       # 5: fig_caption (for correct answer)
                condition_name,    # 6: condition name
                gender,            # 7: gender
                age                # 8: age
            ))
        
        # Process batch when it reaches the specified size
        if len(batch) >= args.batch_size:
            use_json_format = args.use_json_format and not args.no_json_format
            outputs = infer_batch(processor, model, batch, args.device, args.max_new_tokens, args.text_only, use_json_format)
            
            for item, out_text in zip(batch, outputs):
                qid, img_path, qtext, img_rel_item, options_item, fig_caption_item, condition, gender_item, age_item = item
                
                # Get correct answer (for saving ground truth)
                correct_answer = get_correct_answer(fig_caption_item)
                
                if args.print:
                    print(f"Question ID: {qid}")
                    print(f"Condition: {condition}")
                    print(f"Ground truth gender: {gender_item}")
                    print(f"Ground truth age: {age_item}")
                    print(f"Image: {os.path.basename(img_rel_item) if img_rel_item else 'N/A'}")
                    print(f"Model output: {out_text}")
                    print(f"Correct answer: {correct_answer}")
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
            qid, img_path, qtext, img_rel_item, options_item, fig_caption_item, condition, gender_item, age_item = item
            
            correct_answer = get_correct_answer(fig_caption_item)
            
            if args.print:
                print(f"Question ID: {qid}")
                print(f"Condition: {condition}")
                print(f"Ground truth gender: {gender_item}")
                print(f"Ground truth age: {age_item}")
                print(f"Image: {os.path.basename(img_rel_item) if img_rel_item else 'N/A'}")
                print(f"Model output: {out_text}")
                print(f"Correct answer: {correct_answer}")
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
