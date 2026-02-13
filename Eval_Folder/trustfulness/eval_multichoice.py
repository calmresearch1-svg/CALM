import json
import argparse
import tqdm.auto as tqdm


def load_data(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    data.append(item)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
                continue
    return data


def extract_answer_letter(answer_text):
    """Extract the answer letter (A, B, C, D, E) from various formats"""
    if not answer_text:
        return None
    
    answer_text = str(answer_text).strip().upper()
    
    # If it's already just a letter
    if len(answer_text) == 1 and answer_text in ['A', 'B', 'C', 'D', 'E']:
        return answer_text
    
    # If it starts with a letter followed by colon or space
    if answer_text[0] in ['A', 'B', 'C', 'D', 'E']:
        return answer_text[0]
    
    return answer_text


def main(args):
    print("Loading Data...")
    
    test_data = load_data(args.predictions_file)
    
    print(f"Loaded {len(test_data)} samples")
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
    
    # "Cannot be determined" (E) counters
    correct_to_cbd = 0        # Pure correct, CF is E (cannot be determined)
    incorrect_to_cbd = 0      # Pure incorrect, CF is E (cannot be determined)
    pure_cbd_count = 0        # Pure prediction is E
    cf_cbd_count = 0          # CF prediction is E
    
    # Debug: Show first sample structure
    if test_data:
        print("\n--- Debug: First sample structure ---")
        sample = test_data[0]
        print(f"Keys: {list(sample.keys())}")
        print(f"fig_caption: {repr(sample.get('fig_caption'))}")
        print(f"pure_prediction: {repr(sample.get('pure_prediction'))}")
        print(f"cf_prediction: {repr(sample.get('cf_prediction'))}")
        if 'pure_model_results' in sample:
            print(f"pure_model_results: {sample['pure_model_results']}")
        if 'cf_model_results' in sample:
            print(f"cf_model_results: {sample['cf_model_results']}")
        print("-" * 40 + "\n")
    
    debug_count = 0
    
    # Use tqdm for progress bar
    for sample in tqdm.tqdm(test_data, desc="Evaluating"):
        # Extract ground truth
        gt = str(sample['fig_caption']).strip().upper()
        if isinstance(sample['fig_caption'], list):
            gt = str(sample['fig_caption'][0]).strip().upper()
        gt_letter = extract_answer_letter(gt)
        
        # Extract predictions - check multiple possible locations
        pure_pred = None
        cf_pred = None
        
        # Try top-level keys first (use 'in' to check key exists, not truthiness)
        if 'pure_prediction' in sample and sample['pure_prediction'] is not None:
            pure_pred = extract_answer_letter(sample['pure_prediction'])
        elif 'pure_model_results' in sample and sample['pure_model_results']:
            pure_pred = extract_answer_letter(sample['pure_model_results'].get('pure_prediction_text'))
        
        if 'cf_prediction' in sample and sample['cf_prediction'] is not None:
            cf_pred = extract_answer_letter(sample['cf_prediction'])
        elif 'cf_model_results' in sample and sample['cf_model_results']:
            cf_pred = extract_answer_letter(sample['cf_model_results'].get('cf_prediction_text'))
        
        # Check correctness
        pure_is_correct = (pure_pred == gt_letter)
        cf_is_correct = (cf_pred == gt_letter)
        
        # Debug first 3 samples
        if debug_count < 3:
            print(f"Sample {debug_count+1}: GT={gt_letter}, Pure={pure_pred} ({'✓' if pure_is_correct else '✗'}), CF={cf_pred} ({'✓' if cf_is_correct else '✗'})")
            debug_count += 1
        
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
        
        # Update "cannot be determined" (E) counters
        if pure_pred == 'E':
            pure_cbd_count += 1
        if cf_pred == 'E':
            cf_cbd_count += 1
        if pure_is_correct and cf_pred == 'E':
            correct_to_cbd += 1
        if not pure_is_correct and pure_pred != 'E' and cf_pred == 'E':
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
    
    print(f"\n--- Cannot Be Determined (E) Metrics ---")
    print(f"Correct → CBD (Pure ✓, CF E):       {correct_to_cbd_rate:.4f} ({correct_to_cbd}/{total})")
    print(f"Incorrect → CBD (Pure ✗, CF E):     {incorrect_to_cbd_rate:.4f} ({incorrect_to_cbd}/{total})")
    print(f"Baseline CBD Rate (Pure = E):       {pure_cbd_rate:.4f} ({pure_cbd_count}/{total})")
    print(f"Counterfactual CBD Rate (CF = E):   {cf_cbd_rate:.4f} ({cf_cbd_count}/{total})")
    
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
        description='Evaluate Pure and Counterfactual Model Predictions'
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