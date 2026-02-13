"""
ham10000_inference.py

Usage:
    python ham10000_inference.py \\
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \\
        --questions_file counterfactuals_ham10000/ham10000_counterfactual_brain.jsonl \\
        --image_dir /path/to/ham10000_testset \\
        --output_file results/qwen_counterfactual_results.jsonl
"""

import os
import json
import torch
from PIL import Image
import warnings
from tqdm.auto import tqdm
from pathlib import Path
import argparse
from vlm_models import get_model, BaseVLM

warnings.filterwarnings("ignore")

def run_inference(
    model: BaseVLM,
    questions_file: str,
    image_dir: str,
    output_file: str,
    max_samples: int = None
):
    """
    Run inference on HAM10000 counterfactual questions.
    
    Args:
        model: VLM model instance
        questions_file: Path to JSONL file with questions
        image_dir: Directory containing HAM10000 images
        output_file: Path to save results
        max_samples: Maximum number of samples to process (None for all)
    """
    # Load questions
    with open(questions_file, 'r') as f:
        questions = [json.loads(line) for line in f]
    
    if max_samples:
        questions = questions[:max_samples]
    
    print(f"\n{'='*70}")
    print(f"Running inference on {len(questions)} samples")
    print(f"Model: {model.model_id}")
    print(f"Output: {output_file}")
    print('='*70 + '\n')
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    errors = 0
    
    for entry in tqdm(questions, desc="Processing"):
        try:
            # Load image
            image_path = Path(image_dir) / entry['image']
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                errors += 1
                continue
            
            image = Image.open(image_path)
            
            # Get question text
            question = entry['text']
            question = question + "\n Output Format: {'Answer': [A, B, C, D or E], 'Reason': 'Maximum 1-2 sentences.'}." 
            
            # Get model response
            response = model.get_response(image, question)
            
            # Create result entry
            result = {
                'question_id': entry['question_id'],
                'image': entry['image'],
                'question': entry.get('question', entry['text']),
                'options': entry['options'],
                'ground_truth': entry['fig_caption'],
                'response': response,
                'counterfactual_type': entry.get('counterfactual_type', 'original'),
                'expected_answer': entry.get('expected_answer', entry['fig_caption'])
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {entry.get('question_id', 'unknown')}: {e}")
            errors += 1
            continue
    
    # Save results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\n✓ Inference complete!")
    print(f"  Processed: {len(results)}")
    print(f"  Errors: {errors}")
    print(f"  Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on HAM10000 counterfactual questions'
    )
    parser.add_argument(
        '--model_id',
        type=str,
        required=True,
        help='Model ID (e.g., Qwen/Qwen2.5-VL-7B-Instruct)'
    )
    parser.add_argument(
        '--questions_file',
        type=str,
        required=True,
        help='Path to questions JSONL file'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Directory containing HAM10000 images'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to save results'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    print(f"Loading model: {args.model_id}")
    model = get_model(args.model_id, args.device)
    
    # Run inference
    run_inference(
        model=model,
        questions_file=args.questions_file,
        image_dir=args.image_dir,
        output_file=args.output_file,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
