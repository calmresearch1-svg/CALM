"""
iu_xray_inference.py

Pipeline for evaluating VLM robustness using modality/region mismatch detection.
Setting 2: Tests if models can detect when text description doesn't match image.
Uses shared vlm_models.py.
Removed noise injection logic as requested.

Usage:
    python iu_xray_inference.py \\
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \\
        --questions_file /path/to/iu_xray_questions.jsonl \\
        --image_dir /path/to/iu_xray_images \\
        --output_file results/iu_xray_results.jsonl \\
        --cf_type original
"""

import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import warnings
from tqdm.auto import tqdm
import re
from vlm_models import get_model

warnings.filterwarnings("ignore")

# --- Abstention Detection ---

class AbstentionDetector:
    """
    Enhanced abstention detection for VLM responses.
    Detects when model refuses to answer or recognizes mismatch.
    """
    
    def __init__(self):
        # Direct abstention phrases
        self.abstention_phrases = [
            "i don't know", "i do not know", "cannot determine",
            "unable to assess", "cannot assess", "unclear",
            "sorry", "apologize", "i'm not sure", "not certain",
            "cannot answer", "cannot provide", "insufficient information",
            "can't tell", "can not tell", "cannot tell",
            "don't have enough", "do not have enough",
            "not possible to", "impossible to determine",
            "cannot be determined", "can't be determined",
            "abstain"
        ]
        
    def detect_abstention(self, response: str) -> tuple:
        """
        Detect if response contains abstention.
        Returns: (is_abstention, reason)
        """
        response_lower = response.lower().strip()
        
        # Check direct abstention phrases
        for phrase in self.abstention_phrases:
            if phrase in response_lower:
                return True, f"direct_abstention: {phrase}"
        
        # Check for very short non-committal responses
        if len(response.strip()) < 10 and response_lower in ['unclear', 'unknown', 'n/a', 'na']:
            return True, "short_abstention"
        
        return False, "no_abstention"


def extract_yes_no(response: str) -> str:
    """
    Extract yes/no answer from response.
    Returns: 'yes', 'no', or 'abstain'
    """
    response_lower = response.lower().strip()
    
    # Check for abstention first
    detector = AbstentionDetector()
    is_abstain, _ = detector.detect_abstention(response)
    if is_abstain:
        return 'abstain'
    
    # Extract yes/no
    has_yes = 'yes' in response_lower
    has_no = 'no' in response_lower
    
    if has_yes and not has_no:
        return 'yes'
    elif has_no and not has_yes:
        return 'no'
    elif has_yes and has_no:
        # Both present - check which comes first
        yes_idx = response_lower.find('yes')
        no_idx = response_lower.find('no')
        return 'yes' if yes_idx < no_idx else 'no'
    else:
        return 'abstain'


# --- Main Modality Mismatch Evaluation ---

def run_inference(
    model_id: str,
    questions_file: str,
    image_dir: str,
    output_file: str,
    cf_type: str = 'original',
    device: str = 'cuda',
    test_subset: int = None
):
    """
    Main function to run modality/region mismatch evaluation.
    """
    
    if test_subset:
        print(f"TEST MODE: First {test_subset} questions only")
    print("="*70 + "\n")

    # Initialize model
    vlm = get_model(model_id, device=device)
    
    # Load questions
    print(f"Loading questions from: {questions_file}")
    with open(questions_file, 'r') as f:
        questions = [json.loads(line) for line in f]
    
    # Apply test subset if specified
    if test_subset:
        questions = questions[:test_subset]
    
    print(f"✓ Loaded {len(questions)} questions\n")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize detector
    detector = AbstentionDetector()
    
    # Statistics tracking
    stats = {
        'total': 0,
        'abstentions': 0,
        'correct_answers': 0,
        'yes_answers': 0,
        'no_answers': 0,
    }
    
    print("Starting inference...\n")
    
    # Process each question
    with open(output_file, 'w') as outfile:
        for q_idx, item in enumerate(tqdm(questions, desc=f"Processing {cf_type}")):
            
            # Load image
            image_path = os.path.join(image_dir, item['image'])
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}. Skipping.")
                continue
            
            try:
                original_image = Image.open(image_path).convert('RGB')
                # Resize large images
                max_size = 1024
                if max(original_image.size) > max_size:
                    ratio = max_size / max(original_image.size)
                    new_size = tuple(int(dim * ratio) for dim in original_image.size)
                    original_image = original_image.resize(new_size, Image.LANCZOS)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}. Skipping.")
                continue
            
            # Clean question
            question = item['question'].replace('<image>', '').replace('\n', ' ').strip()
            question = question + "\n Output Format: {'Answer': [yes] or [no] or [abstain to answer], 'Reason': 'Maximum 1-2 sentences.'}." 
            ground_truth = item['answer'].strip()
            
            response = vlm.get_response(original_image, question)
            
            # Detect abstention
            is_abstain, abstain_reason = detector.detect_abstention(response)
            
            # Extract answer
            extracted_answer = extract_yes_no(response)
            
            is_correct = None
            gt_clean = ground_truth.lower()
            if 'yes' in gt_clean:
                is_correct = (extracted_answer == 'yes')
            elif 'no' in gt_clean:
                is_correct = (extracted_answer == 'no')
            
            # Update statistics
            stats['total'] += 1
            if is_abstain:
                stats['abstentions'] += 1
            if extracted_answer == 'yes':
                stats['yes_answers'] += 1
            elif extracted_answer == 'no':
                stats['no_answers'] += 1
            if is_correct:
                stats['correct_answers'] += 1
            
            # Save result
            result = {
                'question_index': q_idx,
                'model_name': model_id,
                'cf_type': cf_type,
                'image_path': item['image'],
                
                'question': question,
                'ground_truth': ground_truth,
                'response': response,
                'extracted_answer': extracted_answer,
                
                'is_abstention': is_abstain,
                'abstention_reason': abstain_reason,
                'is_correct': is_correct,
            }
            
            # Add counterfactual metadata if present
            if 'counterfactual_type' in item:
                result['counterfactual_type'] = item['counterfactual_type']
            if 'original_question' in item:
                result['original_question'] = item['original_question']
            
            outfile.write(json.dumps(result) + '\n')
            outfile.flush()
    
    print("\nInference Complete.")
    print(f"Total: {stats['total']}")
    print(f"Abstentions: {stats['abstentions']}")
    if cf_type == 'original':
        print(f"Correct Answers: {stats['correct_answers']}")


import argparse

def main():
    parser = argparse.ArgumentParser(description='Run IU-Xray inference.')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID')
    parser.add_argument('--questions_file', type=str, required=True, help='Path to questions JSONL file')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save results')
    parser.add_argument('--cf_type', type=str, default='original', help='Counterfactual type (e.g., original, cf1_wrong_modality)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--test_subset', type=int, default=None, help='Run on first N questions only')

    args = parser.parse_args()
    
    run_inference(
        model_id=args.model_id,
        questions_file=args.questions_file,
        image_dir=args.image_dir,
        output_file=args.output_file,
        cf_type=args.cf_type,
        device=args.device,
        test_subset=args.test_subset
    )

if __name__ == "__main__":
    main()
