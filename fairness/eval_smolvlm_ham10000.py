#!/usr/bin/env python3
"""
SmolVLM evaluation script for HAM10000 dataset.
Evaluates SmolVLM on multi-choice medical questions from HAM10000 factuality dataset.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


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
    """Write data to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_answer(response: str, debug: bool = False) -> Optional[str]:
    """
    Extract answer from JSON-formatted model response.
    
    Args:
        response: Model's text response containing JSON
        debug: Whether to print debug information
    
    Returns:
        The predicted option letter (A, B, C, or D) or None if unclear
    """
    import json
    import re
    
    if debug:
        print(f"\n=== DEBUG: extract_json_answer ===")
        print(f"Original response: {repr(response)}")
    
    # Try to find JSON in the response
    json_pattern = r'\{[^{}]*"answer"\s*:\s*"([A-D])"[^{}]*\}'
    json_match = re.search(json_pattern, response, re.IGNORECASE | re.DOTALL)
    
    if json_match:
        if debug:
            print(f"Found JSON pattern match: {json_match.group(0)}")
            print(f"Extracted answer: {json_match.group(1)}")
        return json_match.group(1).upper()
    
    # Try to parse the entire response as JSON
    try:
        # Clean up the response to extract just the JSON part
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            if debug:
                print(f"Attempting to parse JSON: {repr(json_str)}")
            
            data = json.loads(json_str)
            if 'answer' in data and data['answer'] in ['A', 'B', 'C', 'D']:
                if debug:
                    print(f"Successfully parsed JSON answer: {data['answer']}")
                return data['answer'].upper()
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        if debug:
            print(f"JSON parsing failed: {e}")
    
    if debug:
        print("No valid JSON answer found")
    
    return None


def extract_multichoice_answer(response: str, options: str, debug: bool = False) -> Optional[str]:
    """
    Extract the most likely answer from model response for multi-choice questions.
    First tries JSON parsing, then falls back to regex patterns.
    
    Args:
        response: Model's text response
        options: String containing options like "A:scalp, B:ear, C:upper extremity, D:lower extremity"
        debug: Whether to print debug information
    
    Returns:
        The predicted option letter (A, B, C, or D) or None if unclear
    """
    if debug:
        print(f"\n=== DEBUG: extract_multichoice_answer ===")
        print(f"Original response: {repr(response)}")
        print(f"Options: {options}")
    
    # FIRST: Try JSON extraction (most reliable)
    json_answer = extract_json_answer(response, debug)
    if json_answer:
        if debug:
            print(f"JSON extraction succeeded: {json_answer}")
        return json_answer
    
    if debug:
        print("JSON extraction failed, falling back to regex patterns...")
    
    # FALLBACK: Use regex patterns (old method)
    response_lower = response.lower().strip()
    
    if debug:
        print(f"Response (lower): {repr(response_lower)}")
    
    # Parse options to extract option letters and descriptions
    option_pattern = r'([A-D]):([^,]+)'
    matches = re.findall(option_pattern, options)
    option_map = {desc.strip().lower(): letter for letter, desc in matches}
    
    if debug:
        print(f"Option map: {option_map}")
    
    # HIGHEST PRIORITY: Look for the specific instruction format "The correct answer is: X" or "The correct answer is: [X]"
    instruction_patterns = [
        r'the correct answer is:\s*\[([A-D])\]',  # "The correct answer is: [D]"
        r'the correct answer is:\s*\[([A-D]):',   # "The correct answer is: [D:"
        r'the correct answer is:\s*([A-D])',      # "The correct answer is: D" (without brackets)
        r'correct answer is:\s*\[([A-D])\]',      # "Correct answer is: [D]"
        r'correct answer is:\s*\[([A-D]):',       # "Correct answer is: [D:"
        r'correct answer is:\s*([A-D])',          # "Correct answer is: D" (without brackets)
    ]
    
    if debug:
        print(f"Testing instruction patterns...")
    
    for i, pattern in enumerate(instruction_patterns):
        match = re.search(pattern, response_lower)
        if debug:
            print(f"  Pattern {i+1}: {pattern} -> {match.group(1) if match else 'No match'}")
        if match:
            result = match.group(1).upper()
            if debug:
                print(f"  FOUND MATCH: {result}")
            return result
    
    # Look for instruction format at the end of the response (most reliable)
    end_instruction_patterns = [
        r'the correct answer is:\s*\[([A-D])\]\.?\s*$',  # "The correct answer is: [D]" at end
        r'the correct answer is:\s*([A-D])\.?\s*$',      # "The correct answer is: D" at end
        r'correct answer is:\s*\[([A-D])\]\.?\s*$',      # "Correct answer is: [D]" at end
        r'correct answer is:\s*([A-D])\.?\s*$',          # "Correct answer is: D" at end
        r'the correct answer is:\s*\[([A-D]):[^\]]*\]\.?\s*$',  # "The correct answer is: [D: description]" at end
    ]
    
    if debug:
        print(f"Testing end instruction patterns...")
    
    for i, pattern in enumerate(end_instruction_patterns):
        match = re.search(pattern, response_lower, re.MULTILINE)
        if debug:
            print(f"  End Pattern {i+1}: {pattern} -> {match.group(1) if match else 'No match'}")
        if match:
            result = match.group(1).upper()
            if debug:
                print(f"  FOUND END MATCH: {result}")
            return result
    
    # Look for other final answer patterns
    final_answer_patterns = [
        r'therefore,?\s*the correct answer is:\s*\[([A-D]):',  # "Therefore, the correct answer is: [D:"
        r'the correct answer is:\s*\[([A-D]):',  # "The correct answer is: [D:"
        r'correct answer:\s*\[([A-D]):',  # "Correct answer: [D:"
        r'answer:\s*\[([A-D]):',  # "Answer: [D:"
        r'final answer:\s*\[([A-D]):',  # "Final answer: [D:"
    ]
    
    if debug:
        print(f"Testing final answer patterns...")
    
    for i, pattern in enumerate(final_answer_patterns):
        match = re.search(pattern, response_lower)
        if debug:
            print(f"  Final Pattern {i+1}: {pattern} -> {match.group(1) if match else 'No match'}")
        if match:
            result = match.group(1).upper()
            if debug:
                print(f"  FOUND FINAL MATCH: {result}")
            return result
    
    # Look for "Answer: X" pattern (without brackets)
    answer_pattern = r'answer:\s*([A-D])'
    answer_match = re.search(answer_pattern, response_lower)
    if debug:
        print(f"Answer pattern: {answer_pattern} -> {answer_match.group(1) if answer_match else 'No match'}")
    if answer_match:
        result = answer_match.group(1).upper()
        if debug:
            print(f"  FOUND ANSWER MATCH: {result}")
        return result
    
    # Look for "The answer is X" pattern
    answer_is_pattern = r'the answer is\s*([A-D])'
    answer_is_match = re.search(answer_is_pattern, response_lower)
    if debug:
        print(f"Answer is pattern: {answer_is_pattern} -> {answer_is_match.group(1) if answer_is_match else 'No match'}")
    if answer_is_match:
        result = answer_is_match.group(1).upper()
        if debug:
            print(f"  FOUND ANSWER IS MATCH: {result}")
        return result
    
    # Look for option mentions in brackets at the end of sentences (common in structured responses)
    bracket_patterns = [
        r'\[([A-D]):[^\]]+\]\.?\s*$',  # [D: melanocytic nevi] at end
        r'\[([A-D]):[^\]]+\]\.?\s*\n',  # [D: melanocytic nevi] at end of line
        r'\[([A-D])\]\.?\s*$',  # [D] at end (just the letter)
    ]
    
    if debug:
        print(f"Testing bracket patterns...")
    
    for i, pattern in enumerate(bracket_patterns):
        match = re.search(pattern, response_lower, re.MULTILINE)
        if debug:
            print(f"  Bracket Pattern {i+1}: {pattern} -> {match.group(1) if match else 'No match'}")
        if match:
            result = match.group(1).upper()
            if debug:
                print(f"  FOUND BRACKET MATCH: {result}")
            return result
    
    # Look for direct option mentions (A, B, C, D) - but be more careful
    # Look for patterns like "A)", "A.", "A:", or standalone "A"
    direct_patterns = [
        r'\b([A-D])\)',  # A), B), C), D)
        r'\b([A-D])\.',  # A., B., C., D.
        r'\b([A-D]):',   # A:, B:, C:, D:
        r'\b([A-D])\s*$',  # A, B, C, D at end of line
    ]
    
    if debug:
        print(f"Testing direct patterns...")
    
    for i, pattern in enumerate(direct_patterns):
        direct_match = re.search(pattern, response_lower)
        if debug:
            print(f"  Direct Pattern {i+1}: {pattern} -> {direct_match.group(1) if direct_match else 'No match'}")
        if direct_match:
            result = direct_match.group(1).upper()
            if debug:
                print(f"  FOUND DIRECT MATCH: {result}")
            return result
    
    # Look for option descriptions in the response
    if debug:
        print(f"Testing option descriptions...")
    
    for desc, letter in option_map.items():
        if desc in response_lower:
            if debug:
                print(f"  FOUND OPTION DESCRIPTION MATCH: {letter} (for '{desc}')")
            return letter.upper()
    
    # Look for partial matches of option descriptions
    if debug:
        print(f"Testing partial option matches...")
    
    for desc, letter in option_map.items():
        desc_words = desc.split()
        if any(word in response_lower for word in desc_words if len(word) > 3):
            if debug:
                print(f"  FOUND PARTIAL MATCH: {letter} (for words from '{desc}')")
            return letter.upper()
    
    # Fallback: look for any standalone A, B, C, D (lowest priority)
    fallback_match = re.search(r'\b([A-D])\b', response_lower)
    if debug:
        print(f"Fallback pattern: \\b([A-D])\\b -> {fallback_match.group(1) if fallback_match else 'No match'}")
    if fallback_match:
        result = fallback_match.group(1).upper()
        if debug:
            print(f"  FOUND FALLBACK MATCH: {result}")
        return result
    
    if debug:
        print(f"NO MATCH FOUND - returning None")
    
    return None


def parse_options(options_str: str) -> List[str]:
    """Parse options string into list of option descriptions."""
    option_pattern = r'([A-D]):([^,]+)'
    matches = re.findall(option_pattern, options_str)
    return [desc.strip() for letter, desc in matches]


def get_correct_answer(fig_caption: str) -> str:
    """Extract the correct answer from fig_caption (e.g., 'C:upper extremity' -> 'C')."""
    match = re.match(r'([A-D]):', fig_caption)
    if match:
        return match.group(1)
    return fig_caption


def build_single_message(user_text: str, include_image: bool, use_json_format: bool = True) -> List[Dict[str, Any]]:
    """Build a single message for SmolVLM chat template."""
    content: List[Dict[str, Any]] = []
    if include_image:
        content.append({"type": "image"})
    
    # Add JSON format instruction for robust answer extraction
    if use_json_format:
        enhanced_text = user_text + "\n\nAnalyze the image and answer the multiple-choice question. Provide your reasoning and the final answer in a JSON format. The JSON object must have two keys: 'reasoning' (a string explaining your logic) and 'answer' (a single uppercase letter corresponding to the correct option)."
    else:
        enhanced_text = user_text
    
    content.append({"type": "text", "text": enhanced_text})
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
    model: AutoModelForVision2Seq,
    batch: List[Tuple],
    device: str,
    max_new_tokens: int,
    text_only: bool,
    use_json_format: bool = True,
) -> List[str]:
    """Run inference on a batch of samples."""
    # batch elements are tuples like (question_id, image, text, image_rel, options, fig_caption)
    messages = [build_single_message(item[2], include_image=(not text_only), use_json_format=use_json_format) for item in batch]
    prompt_texts = [processor.apply_chat_template(m, add_generation_prompt=True) for m in messages]
    
    if text_only:
        inputs = processor(text=prompt_texts, return_tensors="pt", padding=True)
    else:
        images = [item[1] for item in batch]
        # Ensure images are properly loaded and not None
        valid_images = []
        for img in images:
            if img is not None:
                valid_images.append(img)
            else:
                # Create a dummy image if None
                from PIL import Image
                valid_images.append(Image.new('RGB', (224, 224), color='black'))
        
        inputs = processor(text=prompt_texts, images=valid_images, return_tensors="pt", padding=True)
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SmolVLM on HAM10000 factuality JSONL")
    parser.add_argument(
        "--input_jsonl", 
        type=str, 
        default=os.path.join("data", "HAM10000", "HAM10000_factuality.jsonl"), 
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--images_root", 
        type=str, 
        default=os.path.join("data", "HAM10000", "images"), 
        help="Root folder containing images referenced by JSONL"
    )
    parser.add_argument(
        "--output_jsonl", 
        type=str, 
        default=os.path.join("output", "smolvlm", "ham10000_factuality_outputs.jsonl"), 
        help="Path to write model outputs JSONL"
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
        "--load_in_8bit", 
        action="store_true", 
        help="Load model in 8-bit quantization"
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
        help="Print detailed information for each sample (image loading and model output)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Print debug information including raw model outputs and processing steps"
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
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    # Setup model configuration
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    quantization_config = None
    if args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load processor and model
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    # Fix padding side for decoder-only models
    if hasattr(processor, 'tokenizer') and processor.tokenizer:
        processor.tokenizer.padding_side = 'left'
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    attn_impl = "eager"
    if args.device == "cuda" and args.use_flash_attn:
        attn_impl = "flash_attention_2"

    model_kwargs = {
        "quantization_config": quantization_config,
        "_attn_implementation": attn_impl,
    }
    if args.device == "cuda" and not args.load_in_8bit:
        model_kwargs["dtype"] = dtype

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_name,
            **model_kwargs,
        )
    except ImportError as e:
        # Fallback if FlashAttention2 not installed
        if "flash_attn" in str(e).lower() or "flashattention" in str(e).lower():
            model_kwargs["_attn_implementation"] = "eager"
            model = AutoModelForVision2Seq.from_pretrained(
                args.model_name,
                **model_kwargs,
            )
        else:
            raise
    model = model.to(args.device)

    # Load samples
    samples = load_jsonl(args.input_jsonl)
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"Loaded {len(samples)} samples for evaluation")

    rows_for_output: List[Dict[str, Any]] = []
    
    # Track accuracy metrics
    total_samples = 0
    correct_predictions = 0
    unclear_predictions = 0

    batch: List[Tuple] = []
    for sample in tqdm(samples, desc="Evaluating", ncols=100):
        question_id = sample.get("question_id")
        image_rel = sample.get("image")
        text = sample.get("text", "")
        options = sample.get("options", "")
        fig_caption = sample.get("fig_caption", "")

        total_samples += 1

        img = None
        if not args.text_only:
            if image_rel:
                # Handle different image path formats
                image_path = image_rel
                if not os.path.isabs(image_path):
                    image_path = os.path.join(args.images_root, image_rel)
                
                # Try alternative paths if the original doesn't exist
                if not os.path.exists(image_path):
                    # Try with just the filename
                    alt_path = os.path.join(args.images_root, os.path.basename(image_rel))
                    image_path = alt_path if os.path.exists(alt_path) else image_rel
                
                try:
                    img = load_image(image_path)
                    if args.print:
                        print(f"Image loaded: {os.path.basename(image_path)} (size: {img.size})")
                except Exception as e:
                    print(f"Failed to load image {image_path}: {e}")
                    img = None
            
            # If not text_only and image is missing, skip this sample
            if img is None:
                print(f"Skipping sample {question_id} due to missing image")
                continue

        batch.append((question_id, img, text, image_rel, options, fig_caption))
        
        if len(batch) >= args.batch_size:
            use_json_format = args.use_json_format and not args.no_json_format
            outputs = infer_batch(processor, model, batch, args.device, args.max_new_tokens, args.text_only, use_json_format)
            
            for item, out_text in zip(batch, outputs):
                qid, _img, qtext, img_rel_item, options_item, fig_caption_item = item
                
                # Extract predicted answer
                predicted_answer = extract_multichoice_answer(out_text, options_item, args.debug)
                correct_answer = get_correct_answer(fig_caption_item)
                
                # Print detailed information if requested
                if args.print:
                    print(f"Question ID: {qid}")
                    print(f"Image loaded: {os.path.basename(img_rel_item) if img_rel_item else 'N/A'}")
                    print(f"Model output on this image: {predicted_answer}")
                    print(f"Correct answer: {correct_answer}")
                    print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT' if predicted_answer else '? UNCLEAR'}")
                    print(f"Full model response: {out_text}")
                    if args.debug:
                        print(f"Options: {options_item}")
                        print(f"Question text: {qtext}")
                    print("-" * 50)
                
                # Calculate accuracy
                is_correct = predicted_answer == correct_answer if predicted_answer else False
                if predicted_answer is None:
                    unclear_predictions += 1
                elif is_correct:
                    correct_predictions += 1
                
                rows_for_output.append({
                    "question_id": qid,
                    "image": img_rel_item,
                    "text": qtext,
                    "options": options_item,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "model_output": out_text,
                    "is_correct": is_correct,
                    "model": args.model_name,
                })
            
            batch = []

    # Flush remaining batch
    if batch:
        use_json_format = args.use_json_format and not args.no_json_format
        outputs = infer_batch(processor, model, batch, args.device, args.max_new_tokens, args.text_only, use_json_format)
        for item, out_text in zip(batch, outputs):
            qid, _img, qtext, img_rel_item, options_item, fig_caption_item = item
            
            predicted_answer = extract_multichoice_answer(out_text, options_item, args.debug)
            correct_answer = get_correct_answer(fig_caption_item)
            
            # Print detailed information if requested
            if args.print:
                print(f"Question ID: {qid}")
                print(f"Image loaded: {os.path.basename(img_rel_item) if img_rel_item else 'N/A'}")
                print(f"Model output on this image: {predicted_answer}")
                print(f"Correct answer: {correct_answer}")
                print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT' if predicted_answer else '? UNCLEAR'}")
                print(f"Full model response: {out_text}")
                if args.debug:
                    print(f"Options: {options_item}")
                    print(f"Question text: {qtext}")
                print("-" * 50)
            
            is_correct = predicted_answer == correct_answer if predicted_answer else False
            if predicted_answer is None:
                unclear_predictions += 1
            elif is_correct:
                correct_predictions += 1
            
            rows_for_output.append({
                "question_id": qid,
                "image": img_rel_item,
                "text": qtext,
                "options": options_item,
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "model_output": out_text,
                "is_correct": is_correct,
                "model": args.model_name,
            })

    # Write results
    write_jsonl(args.output_jsonl, rows_for_output)

    # Print evaluation results
    evaluated_samples = len(rows_for_output)
    clear_predictions = evaluated_samples - unclear_predictions
    accuracy = correct_predictions / clear_predictions if clear_predictions > 0 else 0.0
    
    print(f"\n=== Evaluation Results ===")
    print(f"Total samples processed: {evaluated_samples}")
    print(f"Clear predictions: {clear_predictions}")
    print(f"Unclear predictions: {unclear_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy (on clear predictions): {accuracy:.4f}")
    print(f"Results saved to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
