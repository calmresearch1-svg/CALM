#!/usr/bin/env python3
"""
Evaluation script for Yes/No/Cannot be determined predictions.
Calculates: Pure accuracy, CF accuracy, Pair accuracy, and Flip rates.
"""

import json
import argparse
from typing import Dict, List


def load_data(filepath: str) -> List[Dict]:
    """Load JSONL data from file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error on line {line_num}: {e}")
    return data


def normalize_answer(answer: str) -> str:
    """
    Normalize answer to: 'yes', 'no', or 'cannot be determined'
    """
    if not answer or not isinstance(answer, str):
        return "cannot be determined"
    
    answer_lower = answer.lower().strip()
    
    if answer_lower.startswith('yes'):
        return "yes"
    elif answer_lower.startswith('no'):
        return "no"
    else:
        return "cannot be determined"


def main(args):
    print("Loading Data...")
    data = load_data(args.predictions_file)
    print(f"Loaded {len(data)} samples")
    print("=" * 80)
    
    # Initialize counters
    pure_correct = 0
    cf_correct = 0
    both_correct = 0
    total = 0
    
    # Flip rate counters
    correct_to_incorrect = 0  # Pure correct, CF incorrect
    incorrect_to_correct = 0  # Pure incorrect, CF correct
    flipped_both_incorrect = 0  # Flipped AND both are incorrect
    total_flipped = 0         # Prediction changed (regardless of correctness)
    
    # "Cannot be determined" counters
    correct_to_cbd = 0        # Pure correct, CF is "cannot be determined"
    incorrect_to_cbd = 0      # Pure incorrect, CF is "cannot be determined"
    pure_cbd_count = 0        # Pure prediction is "cannot be determined"
    cf_cbd_count = 0          # CF prediction is "cannot be determined"
    
    # Debug: show first sample
    if data:
        print("\n--- Debug: First sample structure ---")
        sample = data[0]
        print(f"Keys: {list(sample.keys())}")
        print(f"answer: {repr(sample.get('answer'))}")
        print(f"pure_prediction: {repr(sample.get('pure_prediction'))}")
        print(f"cf_prediction: {repr(sample.get('cf_prediction'))}")
        print("-" * 40 + "\n")
    
    for i, sample in enumerate(data):
        # Extract and normalize ground truth
        gt_raw = sample.get('answer', '')
        gt = normalize_answer(gt_raw)
        
        # Extract and normalize predictions
        pure_pred_raw = sample.get('pure_prediction', '')
        cf_pred_raw = sample.get('cf_prediction', '')
        
        pure_pred = normalize_answer(pure_pred_raw)
        cf_pred = normalize_answer(cf_pred_raw)
        
        # Check correctness
        pure_is_correct = (pure_pred == gt)
        cf_is_correct = (cf_pred == gt)
        
        # Debug first 3 samples
        if i < 3:
            print(f"Sample {i+1}: GT='{gt}', Pure='{pure_pred}' ({'✓' if pure_is_correct else '✗'}), CF='{cf_pred}' ({'✓' if cf_is_correct else '✗'})")
        
        # Update accuracy counters
        if pure_is_correct:
            pure_correct += 1
        if cf_is_correct:
            cf_correct += 1
        if pure_is_correct and cf_is_correct:
            both_correct += 1
        
        # Update flip rate counters
        if pure_is_correct and not cf_is_correct:
            correct_to_incorrect += 1
        if not pure_is_correct and cf_is_correct:
            incorrect_to_correct += 1
        if pure_pred != cf_pred:
            total_flipped += 1
            if not pure_is_correct and not cf_is_correct:
                flipped_both_incorrect += 1
        
        # Update "cannot be determined" counters
        if pure_pred == "cannot be determined":
            pure_cbd_count += 1
        if cf_pred == "cannot be determined":
            cf_cbd_count += 1
        if pure_is_correct and cf_pred == "cannot be determined":
            correct_to_cbd += 1
        if not pure_is_correct and pure_pred != "cannot be determined" and cf_pred == "cannot be determined":
            incorrect_to_cbd += 1
        
        total += 1
    
    # Calculate metrics
    pure_accuracy = pure_correct / total if total > 0 else 0
    cf_accuracy = cf_correct / total if total > 0 else 0
    pair_accuracy = both_correct / total if total > 0 else 0
    
    flip_correct_to_incorrect = correct_to_incorrect / total if total > 0 else 0
    flip_incorrect_to_correct = incorrect_to_correct / total if total > 0 else 0
    flipped_both_incorrect_rate = flipped_both_incorrect / total if total > 0 else 0
    total_flipped_rate = total_flipped / total if total > 0 else 0
    
    # "Cannot be determined" rates
    correct_to_cbd_rate = correct_to_cbd / total if total > 0 else 0
    incorrect_to_cbd_rate = incorrect_to_cbd / total if total > 0 else 0
    pure_cbd_rate = pure_cbd_count / total if total > 0 else 0
    cf_cbd_rate = cf_cbd_count / total if total > 0 else 0
    
    # Print Results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\nTotal Samples Evaluated: {total}")
    
    print(f"\n--- Accuracy Metrics ---")
    print(f"Pure Model Accuracy:          {pure_accuracy:.4f} ({pure_correct}/{total})")
    print(f"Counterfactual Model Accuracy:{cf_accuracy:.4f} ({cf_correct}/{total})")
    print(f"Pair Accuracy (Both Correct): {pair_accuracy:.4f} ({both_correct}/{total})")
    
    print(f"\n--- Flip Rates ---")
    print(f"Correct → Incorrect (Pure ✓, CF ✗): {flip_correct_to_incorrect:.4f} ({correct_to_incorrect}/{total})")
    print(f"Incorrect → Correct (Pure ✗, CF ✓): {flip_incorrect_to_correct:.4f} ({incorrect_to_correct}/{total})")
    print(f"Flipped, Both Wrong (Pure ✗, CF ✗): {flipped_both_incorrect_rate:.4f} ({flipped_both_incorrect}/{total})")
    print(f"Total Flipped (prediction changed): {total_flipped_rate:.4f} ({total_flipped}/{total})")
    
    print(f"\n--- Cannot Be Determined Metrics ---")
    print(f"Correct → CBD (Pure ✓, CF CBD):     {correct_to_cbd_rate:.4f} ({correct_to_cbd}/{total})")
    print(f"Incorrect → CBD (Pure ✗, CF CBD):   {incorrect_to_cbd_rate:.4f} ({incorrect_to_cbd}/{total})")
    print(f"Baseline CBD Rate (Pure = CBD):     {pure_cbd_rate:.4f} ({pure_cbd_count}/{total})")
    print(f"Counterfactual CBD Rate (CF = CBD): {cf_cbd_rate:.4f} ({cf_cbd_count}/{total})")
    
    print("\n" + "=" * 80)
    
    # Save results to JSON if requested
    if args.save_results:
        results = {
            "total_samples": total,
            "accuracy": {
                "pure_accuracy": pure_accuracy,
                "pure_correct": pure_correct,
                "counterfactual_accuracy": cf_accuracy,
                "counterfactual_correct": cf_correct,
                "pair_accuracy": pair_accuracy,
                "both_correct": both_correct
            },
            "flip_rates": {
                "correct_to_incorrect": {
                    "rate": flip_correct_to_incorrect,
                    "count": correct_to_incorrect
                },
                "incorrect_to_correct": {
                    "rate": flip_incorrect_to_correct,
                    "count": incorrect_to_correct
                },
                "flipped_both_incorrect": {
                    "rate": flipped_both_incorrect_rate,
                    "count": flipped_both_incorrect
                },
                "total_flipped": {
                    "rate": total_flipped_rate,
                    "count": total_flipped
                }
            },
            "cannot_be_determined": {
                "correct_to_cbd": {
                    "rate": correct_to_cbd_rate,
                    "count": correct_to_cbd
                },
                "incorrect_to_cbd": {
                    "rate": incorrect_to_cbd_rate,
                    "count": incorrect_to_cbd
                },
                "baseline_cbd_rate": {
                    "rate": pure_cbd_rate,
                    "count": pure_cbd_count
                },
                "counterfactual_cbd_rate": {
                    "rate": cf_cbd_rate,
                    "count": cf_cbd_count
                }
            }
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.save_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate Yes/No/Cannot Be Determined Predictions'
    )
    parser.add_argument(
        '--predictions_file',
        type=str,
        required=True,
        help='Path to the predictions JSONL file'
    )
    parser.add_argument(
        '--save_results',
        type=str,
        default=None,
        help='Path to save aggregated results as JSON file'
    )
    
    args = parser.parse_args()
    main(args)