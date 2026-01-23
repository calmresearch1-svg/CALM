modality_mismatch_pipeline.py
"""
modality_mismatch_pipeline.py

Pipeline for evaluating VLM robustness using modality/region mismatch detection.
Setting 2: Tests if models can detect when text description doesn't match image.

Expected behavior: Model should abstain when detecting mismatch.
"""

import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, Qwen2_5_VLForConditionalGeneration
from skimage.util import random_noise
import numpy as np
import warnings
from tqdm.auto import tqdm
from collections import defaultdict
import re

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# --- 1. Import VLM Handlers from existing pipeline ---
# Reuse your existing BaseVLM, QwenVLM, SmolVLM, get_model, apply_noise functions
# For standalone use, they're included here

class BaseVLM:
    """Abstract base class for Vision Language Models."""
    def __init__(self, model_id, device='cuda'):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._load_model()
        print(f"✓ Model {self.model_id} loaded successfully on {self.device}")

    def _load_model(self):
        """Loads the model and tokenizer. To be implemented by subclasses."""
        raise NotImplementedError

    def get_response(self, image: Image.Image, prompt: str) -> str:
        """Generates a response for a given image and prompt."""
        raise NotImplementedError


class QwenVLM(BaseVLM):
    """Handler for Qwen-VL series models."""
    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.processor.tokenizer.padding_side = 'left'
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16
        ).eval()

    def get_response(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text], 
            images=[image.convert('RGB')], 
            return_tensors="pt"
        ).to(self.device)

        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1
                )
            input_len = inputs['input_ids'].shape[1]
            response_ids = generated_ids[:, input_len:]
            response = self.processor.batch_decode(response_ids, skip_special_tokens=True)[0]
            return response.strip()
        except Exception as e:
            print(f"Error during Qwen inference: {e}")
            return "Error during inference."


class SmolVLM(BaseVLM):
    """Handler for SmolVLM (Idefics3-based)."""
    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'
            
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=self.device
        )

    def get_response(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        
        templated_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=templated_prompt, images=[image], return_tensors="pt").to(self.device)

        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1
                )
            generated_texts = self.processor.batch_decode(
                generated_ids[:, inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            return generated_texts[0].strip()
        except Exception as e:
            print(f"Error during SmolVLM inference: {e}")
            return "Error during inference."


def get_model(model_name: str, device: str = 'cuda'):
    """Factory function to get the appropriate VLM handler."""
    model_map = {
        "qwen": ("Qwen/Qwen2.5-VL-7B-Instruct", QwenVLM),
        "smolvlm": ("HuggingFaceTB/SmolVLM-Instruct", SmolVLM),
    }
    if model_name.lower() in model_map:
        model_id, model_class = model_map[model_name.lower()]
        return model_class(model_id, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")


def apply_noise(image: Image.Image, noise_type: str) -> Image.Image:
    """Applies specified noise to a PIL image."""
    if noise_type == 'original':
        return image
    
    img_array = np.array(image)
    
    if noise_type.startswith('gaussian'):
        std = float(noise_type.split('_')[-1])
        noise = np.random.normal(0, std, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    elif noise_type.startswith('salt_pepper'):
        amount = float(noise_type.split('_')[-1])
        noisy_array_float = random_noise(img_array, mode='s&p', amount=amount)
        noisy_array = (noisy_array_float * 255).astype(np.uint8)
    else:
        print(f"Warning: Unknown noise type '{noise_type}'. Returning original.")
        return image
        
    return Image.fromarray(noisy_array)


# --- 2. Abstention Detection ---

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
        ]
        
        # Mismatch-specific phrases
        self.mismatch_phrases = [
            "not a chest x-ray", "this is not a chest", "not chest",
            "not an x-ray", "not x-ray",
            "appears to be", "seems to be", "looks like",
            "wrong modality", "different modality", "mismatch",
            "ct scan", "mri", "ultrasound", "brain", "abdominal",
            "inconsistent", "contradiction", "doesn't match",
            "not consistent with", "does not match",
            "incorrect modality", "incorrect region",
        ]
        
        # Refusal patterns (regex)
        self.refusal_patterns = [
            r"i (can't|cannot|can not) answer",
            r"unable to (answer|respond|determine)",
            r"(don't|do not) have (enough|sufficient)",
            r"not possible to (answer|determine|assess)",
            r"cannot be (determined|assessed|answered)",
        ]
    
    def detect_abstention(self, response: str) -> tuple:
        """
        Detect if response contains abstention.
        Returns: (is_abstention, reason, mismatch_detected)
        """
        response_lower = response.lower().strip()
        
        # Check direct abstention phrases
        for phrase in self.abstention_phrases:
            if phrase in response_lower:
                return True, f"direct_abstention: {phrase}", False
        
        # Check mismatch detection phrases (special case - also counts as abstention)
        for phrase in self.mismatch_phrases:
            if phrase in response_lower:
                return True, f"mismatch_detected: {phrase}", True
        
        # Check regex patterns
        for pattern in self.refusal_patterns:
            if re.search(pattern, response_lower):
                return True, f"refusal_pattern: {pattern}", False
        
        # Check for very short non-committal responses
        if len(response.strip()) < 10 and response_lower in ['unclear', 'unknown', 'n/a', 'na']:
            return True, "short_abstention", False
        
        return False, "no_abstention", False


def extract_yes_no(response: str) -> str:
    """
    Extract yes/no answer from response.
    Returns: 'yes', 'no', or 'abstain'
    """
    response_lower = response.lower().strip()
    
    # Check for abstention first
    detector = AbstentionDetector()
    is_abstain, _, _ = detector.detect_abstention(response)
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


# --- 3. Main Modality Mismatch Evaluation ---

def run_modality_mismatch_evaluation(config: dict):
    """
    Main function to run modality/region mismatch evaluation.
    
    Args:
        config (dict): Configuration with keys:
            - model_name: 'qwen' or 'smolvlm'
            - data_path: Path to counterfactual JSONL (e.g., iuxray_cf1_wrong_modality.jsonl)
            - image_base_path: Base directory for images
            - output_path: Path to save results
            - cf_type: 'original', 'cf1_wrong_modality', 'cf2_wrong_region', 'cf3_both_wrong'
            - noise_types: List of noise types (default: ['original'])
            - device: 'cuda' or 'cpu'
            - test_subset: Optional int to test on first N questions only
    """
    print("\n" + "="*70)
    print("MODALITY/REGION MISMATCH EVALUATION - SETTING 2")
    print("="*70)
    print(f"Model: {config['model_name']}")
    print(f"Condition: {config['cf_type']}")
    print(f"Noise types: {config['noise_types']}")
    print(f"Device: {config.get('device', 'cuda')}")
    if config.get('test_subset'):
        print(f"⚠️  TEST MODE: First {config['test_subset']} questions only")
    print("="*70 + "\n")

    # Initialize model
    vlm = get_model(config['model_name'], device=config.get('device', 'cuda'))
    
    # Load questions
    print(f"Loading questions from: {config['data_path']}")
    with open(config['data_path'], 'r') as f:
        questions = [json.loads(line) for line in f]
    
    # Apply test subset if specified
    if config.get('test_subset'):
        questions = questions[:config['test_subset']]
    
    print(f"✓ Loaded {len(questions)} questions\n")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    
    # Initialize detector
    detector = AbstentionDetector()
    
    # Statistics tracking
    stats = {noise: {
        'total': 0,
        'abstentions': 0,
        'mismatch_detected': 0,
        'correct_answers': 0,  # For original questions only
        'yes_answers': 0,
        'no_answers': 0,
    } for noise in config['noise_types']}
    
    print("Starting inference...\n")
    
    # Process each question
    with open(config['output_path'], 'w') as outfile:
        for q_idx, item in enumerate(tqdm(questions, desc=f"Processing {config['cf_type']}")):
            
            # Load image
            image_path = os.path.join(config['image_base_path'], item['image'])
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
            ground_truth = item['answer'].strip()
            
            # Test on each noise type
            for noise_type in config['noise_types']:
                noisy_image = apply_noise(original_image, noise_type)
                
                # Get model response
                response = vlm.get_response(noisy_image, question)
                
                # Detect abstention
                is_abstain, abstain_reason, mismatch_detected = detector.detect_abstention(response)
                
                # Extract answer
                extracted_answer = extract_yes_no(response)
                
                # Evaluate correctness (for original questions only)
                is_correct = None
                if config['cf_type'] == 'original':
                    gt_clean = ground_truth.lower()
                    if 'yes' in gt_clean:
                        is_correct = (extracted_answer == 'yes')
                    elif 'no' in gt_clean:
                        is_correct = (extracted_answer == 'no')
                
                # Update statistics
                stats[noise_type]['total'] += 1
                if is_abstain:
                    stats[noise_type]['abstentions'] += 1
                if mismatch_detected:
                    stats[noise_type]['mismatch_detected'] += 1
                if extracted_answer == 'yes':
                    stats[noise_type]['yes_answers'] += 1
                elif extracted_answer == 'no':
                    stats[noise_type]['no_answers'] += 1
                if is_correct:
                    stats[noise_type]['correct_answers'] += 1
                
                # Save result
                result = {
                    'question_index': q_idx,
                    'noise_type': noise_type,
                    'model_name': config['model_name'],
                    'cf_type': config['cf_type'],
                    'image_path': item['image'],
                    
                    'question': question,
                    'ground_truth': ground_truth,
                    'response': response,
                    'extracted_answer': extracted_answer,
                    
                    'is_abstention': is_abstain,
                    'abstention_reason': abstain_reason,
                    'mismatch_detected': mismatch_detected,
                    'is_correct': is_correct,
                }
                
                # Add counterfactual metadata if present
                if 'counterfactual_type' in item:
                    result['counterfactual_type'] = item['counterfactual_type']
                if 'original_question' in item:
                    result['original_question'] = item['original_question']
                
                outfile.write(json.dumps(result) + '\n')
                outfile.flush()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    for noise_type in config['noise_types']:
        s = stats[noise_type]
        print(f"\n{noise_type.upper()}:")
        print("-" * 70)
        print(f"Total questions: {s['total']}")
        
        # Abstention metrics
        abs_rate = s['abstentions'] / s['total'] * 100 if s['total'] > 0 else 0
        mismatch_rate = s['mismatch_detected'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f"Abstention Rate: {abs_rate:.2f}% ({s['abstentions']}/{s['total']})")
        print(f"Mismatch Detection Rate: {mismatch_rate:.2f}% ({s['mismatch_detected']}/{s['total']})")
        
        # Answer distribution
        print(f"\nAnswer Distribution:")
        print(f"  Yes: {s['yes_answers']} ({s['yes_answers']/s['total']*100:.1f}%)")
        print(f"  No: {s['no_answers']} ({s['no_answers']/s['total']*100:.1f}%)")
        print(f"  Abstain: {s['abstentions']} ({abs_rate:.1f}%)")
        
        # Accuracy (for original only)
        if config['cf_type'] == 'original' and s['correct_answers'] > 0:
            answered = s['total'] - s['abstentions']
            accuracy = s['correct_answers'] / s['total'] * 100
            accuracy_answered = s['correct_answers'] / answered * 100 if answered > 0 else 0
            print(f"\nAccuracy (all): {accuracy:.2f}%")
            print(f"Accuracy (answered only): {accuracy_answered:.2f}%")
    
    print("\n" + "="*70)
    print(f"✓ Results saved to: {config['output_path']}")
    print("="*70 + "\n")


# --- 4. Results Analysis ---

def analyze_mismatch_results(results_paths: dict):
    """
    Compare results across different counterfactual conditions.
    
    Args:
        results_paths: Dict mapping condition names to result file paths
            e.g., {'original': 'path1.jsonl', 'cf1_wrong_modality': 'path2.jsonl', ...}
    """
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS - MODALITY MISMATCH ROBUSTNESS")
    print("="*70)
    
    all_results = {}
    for condition, path in results_paths.items():
        with open(path, 'r') as f:
            all_results[condition] = [json.loads(line) for line in f]
    
    # Group by noise type
    noise_types = set()
    for results in all_results.values():
        noise_types.update(r['noise_type'] for r in results)
    
    for noise_type in sorted(noise_types):
        print(f"\n{noise_type.upper()}:")
        print("-" * 70)
        print(f"{'Condition':<25} {'Abstention %':<15} {'Mismatch %':<15} {'Accuracy %':<15}")
        print("-" * 70)
        
        for condition in ['original', 'cf1_wrong_modality', 'cf2_wrong_region', 'cf3_both_wrong']:
            if condition not in all_results:
                continue
            
            results = [r for r in all_results[condition] if r['noise_type'] == noise_type]
            if not results:
                continue
            
            total = len(results)
            abstentions = sum(1 for r in results if r['is_abstention'])
            mismatch = sum(1 for r in results if r['mismatch_detected'])
            correct = sum(1 for r in results if r.get('is_correct'))
            
            abs_rate = abstentions / total * 100
            mismatch_rate = mismatch / total * 100
            acc_rate = correct / total * 100 if condition == 'original' else 0
            
            acc_str = f"{acc_rate:.2f}%" if condition == 'original' else "N/A"
            print(f"{condition:<25} {abs_rate:>6.2f}%{'':<8} {mismatch_rate:>6.2f}%{'':<8} {acc_str:<15}")
    
    # Expected behavior summary
    print("\n" + "="*70)
    print("ROBUSTNESS SUMMARY")
    print("="*70)
    print("\nExpected Behavior:")
    print("  • Original: LOW abstention (~2-5%), HIGH accuracy")
    print("  • CF1/CF2: MEDIUM-HIGH abstention (~40-80%)")
    print("  • CF3: VERY HIGH abstention (~70-90%)")
    print("\nKey Metric: Δ Abstention (Counterfactual - Original)")
    print("  Higher Δ = Better mismatch detection = More robust model")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("Modality/Region Mismatch Evaluation Pipeline - Setting 2")
    print("="*70)
    print("\nThis pipeline evaluates VLM robustness to modality/region mismatches.")
    print("\nUsage:")
    print("  1. Generate counterfactuals using the generator script")
    print("  2. Configure paths in run script")
    print("  3. Run evaluation for each condition")
    print("  4. Use analyze_mismatch_results() to compare")
    print("="*70)
