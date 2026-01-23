#!/usr/bin/env python3
"""
Counterfactual Fairness Experiment Script for SmolVLM on FairVLM Dataset.
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
from transformers import AutoModelForVision2Seq, AutoProcessor


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
    """Load question IDs that have already been processed from a JSONL file."""
    processed = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        processed.add(row.get("question_id"))
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


def get_sample_pairs(data_dir: str) -> List[Tuple[int, str, str]]:
    """
    Find all valid npz-jpg pairs in the data directory.
    
    Returns:
        List of tuples (question_id, npz_path, jpg_path)
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
        
        sample_id_str = match.group(1)
        question_id = int(sample_id_str)  # Convert to integer
        
        # Find corresponding image (still use string for filename)
        jpg_path = os.path.join(data_dir, f"slo_fundus_{sample_id_str}.jpg")
        
        if os.path.exists(jpg_path):
            pairs.append((question_id, npz_path, jpg_path))
        else:
            print(f"Warning: Image not found for sample {sample_id_str}: {jpg_path}")
    
    return pairs


def build_smolvlm_message(user_text: str, use_json_format: bool = True) -> List[Dict[str, Any]]:
    """Build a message for SmolVLM chat template."""
    # Add JSON format instruction for robust answer extraction
    if use_json_format:
        enhanced_text = user_text + "\n\nAnalyze the image and answer the question. Provide your reasoning and the final answer in a JSON format. The JSON object must have two keys: 'reasoning' (a string explaining your logic) and 'answer' (a string, either 'yes' or 'no')."
    else:
        enhanced_text = user_text
    
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": enhanced_text}
            ],
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
    model: AutoModelForVision2Seq,
    batch: List[Tuple],
    device: str,
    max_new_tokens: int,
    text_only: bool,
    use_json_format: bool = True,
) -> List[str]:
    """
    Run inference on a batch of samples using SmolVLM.
    
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
        # Text-only inference (no images)
        texts = []
        for item in batch:
            _, _, text, _, _, _, _, _, _, _ = item
            messages = build_smolvlm_message(text, use_json_format=use_json_format)
            templated_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(templated_prompt)
        
        inputs = processor(text=texts, return_tensors="pt", padding=True)
    else:
        # Multi-modal inference
        texts = []
        images = []
        
        for item in batch:
            _, image_path, text, _, _, _, _, _, _, _ = item
            
            # Load and prepare image
            try:
                image = Image.open(image_path).convert('RGB')
                images.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Use a blank image as fallback
                images.append(Image.new('RGB', (224, 224), color='white'))
            
            # Build message and apply chat template
            messages = build_smolvlm_message(text, use_json_format=use_json_format)
            templated_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(templated_prompt)
        
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
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
    parser = argparse.ArgumentParser(description="Counterfactual Fairness Experiment for SmolVLM on FairVLM Dataset")
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
        default="fairvlm_smolvlm_baseline.jsonl", 
        help="Filename for baseline outputs"
    )
    parser.add_argument(
        "--output_counterfactual", 
        type=str, 
        default="fairvlm_smolvlm_counterfactual.jsonl", 
        help="Filename for counterfactual outputs"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="HuggingFaceTB/SmolVLM-Instruct", 
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

    print(f"Loading SmolVLM model: {args.model_name}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Set padding side to left for decoder-only models
    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.padding_side = 'left'

    # Load model - SmolVLM uses AutoModelForVision2Seq
    print(f"Loading model with dtype: {args.dtype}")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if args.device == "cuda" else None,
    )
    
    if args.device == "cpu":
        model = model.to(args.device)
    
    model.eval()
    print(f"Model loaded successfully on {args.device}")

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
    
    for question_id, npz_path, jpg_path in tqdm(sample_pairs, desc="Running experiment", ncols=100):
        # Skip already processed samples in resume mode
        if question_id in already_processed:
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
                question_id,         # 0: question_id
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
                qid, img_path, qtext, condition, gender, age, race, glaucoma, gender_code, race_code = item
                
                if args.print:
                    print(f"Question ID: {qid}")
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
                    "question_id": qid,
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
            qid, img_path, qtext, condition, gender, age, race, glaucoma, gender_code, race_code = item
            
            if args.print:
                print(f"Question ID: {qid}")
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
                "question_id": qid,
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

