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
    """Extract the answer letter (A, B, C, D) from various formats"""
    if not answer_text:
        return None
    
    answer_text = str(answer_text).strip().upper()
    
    # If it's already just a letter
    if len(answer_text) == 1 and answer_text in ['A', 'B', 'C', 'D']:
        return answer_text
    
    # If it starts with a letter followed by colon or space
    if answer_text[0] in ['A', 'B', 'C', 'D']:
        return answer_text[0]
    
    return answer_text


def main(args):
    print("Loading Data...")
    
    test_data = load_data(args.predictions_file)
    
    print(f"Loaded {len(test_data)} samples")
    print("=" * 80)

    # Initialize metrics
    pure_correct = 0
    cf_correct = 0
    both_correct = 0
    total = 0
    
    # Entropy statistics
    pure_full_entropies = []
    pure_answer_entropies = []
    cf_full_entropies = []
    cf_answer_entropies = []
    
    # For correct vs incorrect analysis
    pure_correct_full_entropy = []
    pure_incorrect_full_entropy = []
    pure_correct_answer_entropy = []
    pure_incorrect_answer_entropy = []
    
    cf_correct_full_entropy = []
    cf_incorrect_full_entropy = []
    cf_correct_answer_entropy = []
    cf_incorrect_answer_entropy = []
    
    # Use tqdm for progress bar
    for sample in tqdm.tqdm(test_data, desc="Evaluating"):
        # Extract ground truths
        pure_gt = str(sample['fig_caption']).strip().upper()
        if isinstance(sample['fig_caption'], list):
            pure_gt = str(sample['fig_caption'][0]).strip().upper()
        pure_gt_letter = extract_answer_letter(pure_gt)
        
        cf_gt = "D"  # Counterfactual ground truth is always D (None of the above)
        
        # Extract predictions
        pure_pred = extract_answer_letter(sample['pure_model_results']['pure_prediction_text'])
        cf_pred = extract_answer_letter(sample['cf_model_results']['cf_prediction_text'])
        
        # Check correctness
        pure_is_correct = (pure_pred == pure_gt_letter)
        cf_is_correct = (cf_pred == cf_gt)
        
        if pure_is_correct:
            pure_correct += 1
        if cf_is_correct:
            cf_correct += 1
        if pure_is_correct and cf_is_correct:
            both_correct += 1
        
        total += 1
        
        # Collect entropy values
        if 'pure_uncertainty_full' in sample['pure_model_results']:
            pure_full = sample['pure_model_results']['pure_uncertainty_full']
            pure_full_entropies.append(pure_full)
            
            if pure_is_correct:
                pure_correct_full_entropy.append(pure_full)
            else:
                pure_incorrect_full_entropy.append(pure_full)
        
        if 'pure_uncertainty_answer' in sample['pure_model_results']:
            pure_answer = sample['pure_model_results']['pure_uncertainty_answer']
            pure_answer_entropies.append(pure_answer)
            
            if pure_is_correct:
                pure_correct_answer_entropy.append(pure_answer)
            else:
                pure_incorrect_answer_entropy.append(pure_answer)
        
        if 'cf_uncertainty_full' in sample['cf_model_results']:
            cf_full = sample['cf_model_results']['cf_uncertainty_full']
            cf_full_entropies.append(cf_full)
            
            if cf_is_correct:
                cf_correct_full_entropy.append(cf_full)
            else:
                cf_incorrect_full_entropy.append(cf_full)
        
        if 'cf_uncertainty_answer' in sample['cf_model_results']:
            cf_answer = sample['cf_model_results']['cf_uncertainty_answer']
            cf_answer_entropies.append(cf_answer)
            
            if cf_is_correct:
                cf_correct_answer_entropy.append(cf_answer)
            else:
                cf_incorrect_answer_entropy.append(cf_answer)

    # Calculate and Print Metrics
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    pure_accuracy = pure_correct / total if total > 0 else 0
    cf_accuracy = cf_correct / total if total > 0 else 0
    both_accuracy = both_correct / total if total > 0 else 0
    
    print(f"\nTotal Samples Evaluated: {total}")
    print(f"\n--- Accuracy Metrics ---")
    print(f"Pure Model Accuracy:           {pure_accuracy:.6f} ({pure_correct}/{total})")
    print(f"Counterfactual Model Accuracy: {cf_accuracy:.6f} ({cf_correct}/{total})")
    print(f"Both Correct:                  {both_accuracy:.6f} ({both_correct}/{total})")
    print(f"Both Correct Percentage:       {both_accuracy * 100:.2f}%")
    
    # Entropy analysis
    if pure_full_entropies or pure_answer_entropies or cf_full_entropies or cf_answer_entropies:
        import statistics
        
        print(f"\n" + "=" * 80)
        print("PREDICTIVE ENTROPY STATISTICS")
        print("=" * 80)
        
        # Pure Model - Full Entropy
        if pure_full_entropies:
            print(f"\n--- Pure Model - Full Predictive Entropy ---")
            print(f"  Mean:   {statistics.mean(pure_full_entropies):.6f}")
            print(f"  Median: {statistics.median(pure_full_entropies):.6f}")
            print(f"  StdDev: {statistics.stdev(pure_full_entropies) if len(pure_full_entropies) > 1 else 0:.6f}")
            print(f"  Min:    {min(pure_full_entropies):.6f}")
            print(f"  Max:    {max(pure_full_entropies):.6f}")
        
        # Pure Model - Answer Entropy
        if pure_answer_entropies:
            print(f"\n--- Pure Model - Answer Predictive Entropy ---")
            print(f"  Mean:   {statistics.mean(pure_answer_entropies):.6f}")
            print(f"  Median: {statistics.median(pure_answer_entropies):.6f}")
            print(f"  StdDev: {statistics.stdev(pure_answer_entropies) if len(pure_answer_entropies) > 1 else 0:.6f}")
            print(f"  Min:    {min(pure_answer_entropies):.6f}")
            print(f"  Max:    {max(pure_answer_entropies):.6f}")
        
        # Counterfactual Model - Full Entropy
        if cf_full_entropies:
            print(f"\n--- Counterfactual Model - Full Predictive Entropy ---")
            print(f"  Mean:   {statistics.mean(cf_full_entropies):.6f}")
            print(f"  Median: {statistics.median(cf_full_entropies):.6f}")
            print(f"  StdDev: {statistics.stdev(cf_full_entropies) if len(cf_full_entropies) > 1 else 0:.6f}")
            print(f"  Min:    {min(cf_full_entropies):.6f}")
            print(f"  Max:    {max(cf_full_entropies):.6f}")
        
        # Counterfactual Model - Answer Entropy
        if cf_answer_entropies:
            print(f"\n--- Counterfactual Model - Answer Predictive Entropy ---")
            print(f"  Mean:   {statistics.mean(cf_answer_entropies):.6f}")
            print(f"  Median: {statistics.median(cf_answer_entropies):.6f}")
            print(f"  StdDev: {statistics.stdev(cf_answer_entropies) if len(cf_answer_entropies) > 1 else 0:.6f}")
            print(f"  Min:    {min(cf_answer_entropies):.6f}")
            print(f"  Max:    {max(cf_answer_entropies):.6f}")
        
        # Comparison: Pure vs Counterfactual
        if pure_full_entropies and cf_full_entropies:
            print(f"\n--- Entropy Comparison (Pure vs Counterfactual) ---")
            print(f"Full Entropy Difference (Pure - CF):   {statistics.mean(pure_full_entropies) - statistics.mean(cf_full_entropies):.6f}")
            if pure_answer_entropies and cf_answer_entropies:
                print(f"Answer Entropy Difference (Pure - CF): {statistics.mean(pure_answer_entropies) - statistics.mean(cf_answer_entropies):.6f}")
        
        # Correct vs Incorrect Analysis
        if args.show_detailed:
            print(f"\n" + "=" * 80)
            print("ENTROPY ANALYSIS: CORRECT VS INCORRECT PREDICTIONS")
            print("=" * 80)
            
            # Pure Model
            if pure_correct_full_entropy and pure_incorrect_full_entropy:
                print(f"\n--- Pure Model - Full Entropy ---")
                print(f"Correct Predictions:   {statistics.mean(pure_correct_full_entropy):.6f} (n={len(pure_correct_full_entropy)})")
                print(f"Incorrect Predictions: {statistics.mean(pure_incorrect_full_entropy):.6f} (n={len(pure_incorrect_full_entropy)})")
                print(f"Difference:            {statistics.mean(pure_incorrect_full_entropy) - statistics.mean(pure_correct_full_entropy):.6f}")
            
            if pure_correct_answer_entropy and pure_incorrect_answer_entropy:
                print(f"\n--- Pure Model - Answer Entropy ---")
                print(f"Correct Predictions:   {statistics.mean(pure_correct_answer_entropy):.6f} (n={len(pure_correct_answer_entropy)})")
                print(f"Incorrect Predictions: {statistics.mean(pure_incorrect_answer_entropy):.6f} (n={len(pure_incorrect_answer_entropy)})")
                print(f"Difference:            {statistics.mean(pure_incorrect_answer_entropy) - statistics.mean(pure_correct_answer_entropy):.6f}")
            
            # Counterfactual Model
            if cf_correct_full_entropy and cf_incorrect_full_entropy:
                print(f"\n--- Counterfactual Model - Full Entropy ---")
                print(f"Correct Predictions:   {statistics.mean(cf_correct_full_entropy):.6f} (n={len(cf_correct_full_entropy)})")
                print(f"Incorrect Predictions: {statistics.mean(cf_incorrect_full_entropy):.6f} (n={len(cf_incorrect_full_entropy)})")
                print(f"Difference:            {statistics.mean(cf_incorrect_full_entropy) - statistics.mean(cf_correct_full_entropy):.6f}")
            
            if cf_correct_answer_entropy and cf_incorrect_answer_entropy:
                print(f"\n--- Counterfactual Model - Answer Entropy ---")
                print(f"Correct Predictions:   {statistics.mean(cf_correct_answer_entropy):.6f} (n={len(cf_correct_answer_entropy)})")
                print(f"Incorrect Predictions: {statistics.mean(cf_incorrect_answer_entropy):.6f} (n={len(cf_incorrect_answer_entropy)})")
                print(f"Difference:            {statistics.mean(cf_incorrect_answer_entropy) - statistics.mean(cf_correct_answer_entropy):.6f}")
    
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
                "both_correct": both_correct,
                "both_correct_percentage": both_accuracy * 100
            },
            "entropy": {
                "pure_full": {
                    "mean": statistics.mean(pure_full_entropies) if pure_full_entropies else None,
                    "median": statistics.median(pure_full_entropies) if pure_full_entropies else None,
                    "std": statistics.stdev(pure_full_entropies) if len(pure_full_entropies) > 1 else None,
                    "min": min(pure_full_entropies) if pure_full_entropies else None,
                    "max": max(pure_full_entropies) if pure_full_entropies else None
                },
                "pure_answer": {
                    "mean": statistics.mean(pure_answer_entropies) if pure_answer_entropies else None,
                    "median": statistics.median(pure_answer_entropies) if pure_answer_entropies else None,
                    "std": statistics.stdev(pure_answer_entropies) if len(pure_answer_entropies) > 1 else None,
                    "min": min(pure_answer_entropies) if pure_answer_entropies else None,
                    "max": max(pure_answer_entropies) if pure_answer_entropies else None
                },
                "counterfactual_full": {
                    "mean": statistics.mean(cf_full_entropies) if cf_full_entropies else None,
                    "median": statistics.median(cf_full_entropies) if cf_full_entropies else None,
                    "std": statistics.stdev(cf_full_entropies) if len(cf_full_entropies) > 1 else None,
                    "min": min(cf_full_entropies) if cf_full_entropies else None,
                    "max": max(cf_full_entropies) if cf_full_entropies else None
                },
                "counterfactual_answer": {
                    "mean": statistics.mean(cf_answer_entropies) if cf_answer_entropies else None,
                    "median": statistics.median(cf_answer_entropies) if cf_answer_entropies else None,
                    "std": statistics.stdev(cf_answer_entropies) if len(cf_answer_entropies) > 1 else None,
                    "min": min(cf_answer_entropies) if cf_answer_entropies else None,
                    "max": max(cf_answer_entropies) if cf_answer_entropies else None
                }
            }
        }
        
        output_file = args.save_results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate Pure and Counterfactual Model Predictions'
    )
    parser.add_argument(
        '--predictions_file', 
        type=str, 
        required=True, 
        help='Path to the predictions JSONL file containing both pure and counterfactual results'
    )
    parser.add_argument(
        '--show_detailed', 
        action='store_true',
        help='Show detailed entropy analysis for correct vs incorrect predictions'
    )
    parser.add_argument(
        '--save_results',
        type=str,
        default=None,
        help='Path to save aggregated results as JSON file'
    )

    args = parser.parse_args()
    main(args)