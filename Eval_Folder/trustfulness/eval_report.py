import json
import math
import sys
import numpy as np

def clean_labels(data):
    """
    Converts 'null' (which becomes None in Python) to float('nan') 
    in all label dictionaries for a single data entry.
    This is critical for the math operations.
    """
    label_dicts_to_clean = []
    
    if data.get('ground_truth_labels'):
        label_dicts_to_clean.append(data['ground_truth_labels'])
        
    if data.get('pure_prediction') and data['pure_prediction'].get('chexpert_labels'):
        label_dicts_to_clean.append(data['pure_prediction']['chexpert_labels'])
        
    if data.get('counterfactual_prediction') and data['counterfactual_prediction'].get('chexpert_labels'):
        label_dicts_to_clean.append(data['counterfactual_prediction']['chexpert_labels'])

    for label_dict in label_dicts_to_clean:
        if label_dict:
            for key, value in label_dict.items():
                if value is None:
                    label_dict[key] = float('nan')
    return data

def calculate_chair(pred_labels, gt_labels, all_labels):
    """
    Calculates the CHAIR score: (False Positives) / (All Mentioned in Prediction)
    
    - False Positive: Prediction is 1.0 (positive) when Ground Truth is 0.0 (negative).
    - Mentioned in Prediction: Any label in the prediction that is not NaN (i.e., 1.0, 0.0, or -1.0).
    """
    false_positives = 0
    mentioned_in_pred = 0
    
    if not pred_labels:
        return 0.0
    
    for label in all_labels:
        pred_l = pred_labels.get(label, float('nan'))
        gt_l = gt_labels.get(label, float('nan'))
        
        # Check if mentioned in prediction (not NaN)
        if not math.isnan(pred_l):
            mentioned_in_pred += 1
            
        # Check for False Positive (Pred=1, GT=0)
        if pred_l == 1.0 and gt_l == 0.0:
            false_positives += 1
            
    # Avoid division by zero
    if mentioned_in_pred == 0:
        return 0.0
        
    return false_positives / mentioned_in_pred

def calculate_accuracy_and_ratios(pred_labels, gt_labels, all_labels):
    """
    Calculates:
    1. Accuracy (without uncertain): (Correct Matches on 0/1) / (All GT labels that are 0.0 or 1.0)
    2. Accuracy (with uncertain): (Correct Matches on 0/1/-1) / (All GT labels that are 0.0, 1.0, or -1.0)
    3. Uncertainty Ratio: (Pred is -1.0 when GT is 0.0/1.0) / (All GT labels that are 0.0 or 1.0)
    4. Overconfidence Ratio: (Pred is 0.0/1.0 when GT is -1.0) / (All GT labels that are -1.0)
    """
    correct_matches = 0  # Correct on 0/1 only
    total_gt = 0  # Count of GT that are 0/1
    
    correct_matches_unc = 0  # Correct on -1 only
    total_uncertain_gt = 0  # Count of GT that are -1
    
    uncertain_events = 0  # Pred=-1 when GT=0/1
    overconfident_events = 0  # Pred=0/1 when GT=-1
    
    if not pred_labels or not gt_labels:
        return {
            "accuracy": 0.0,
            "accuracy_wth_unc": 0.0,
            "uncertainty_ratio": 0.0,
            "overconfidence_ratio": 0.0
        }
    
    for label in all_labels:
        pred_l = pred_labels.get(label, float('nan'))
        gt_l = gt_labels.get(label, float('nan'))

        # Case 1: GT is certain (0.0 or 1.0)
        if gt_l == 0.0 or gt_l == 1.0:
            total_gt += 1
            
            # Check for correct match
            if pred_l == gt_l:
                correct_matches += 1
            
            # Check for uncertainty event (pred=-1 when GT is certain)
            if pred_l == -1.0:
                uncertain_events += 1
        
        # Case 2: GT is uncertain (-1.0)
        elif gt_l == -1.0:
            total_uncertain_gt += 1
            
            # Check for correct match on uncertain
            if pred_l == gt_l:
                correct_matches_unc += 1
            
            # Check for overconfidence event (pred=0/1 when GT is uncertain)
            if pred_l == 0.0 or pred_l == 1.0:
                overconfident_events += 1

    # Calculate final scores, handling division by zero
    accuracy = (correct_matches / total_gt) if total_gt > 0 else 0.0
    accuracy_wth_unc = ((correct_matches + correct_matches_unc) / (total_gt + total_uncertain_gt)) if (total_gt + total_uncertain_gt) > 0 else 0.0
    uncertainty_ratio = (uncertain_events / total_gt) if total_gt > 0 else 0.0
    overconfidence_ratio = (overconfident_events / total_uncertain_gt) if total_uncertain_gt > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "accuracy_wth_unc": accuracy_wth_unc,
        "uncertainty_ratio": uncertainty_ratio,
        "overconfidence_ratio": overconfidence_ratio
    }

def calculate_trigger_faithfulness(cf_prediction):
    """
    Checks if the counterfactual prediction label matches the trigger finding.
    Returns 1.0 if faithful (trigger was predicted as 1.0), 0.0 otherwise.
    """
    if not cf_prediction or 'trigger_finding' not in cf_prediction:
        return 0.0
        
    trigger = cf_prediction.get('trigger_finding')
    cf_labels = cf_prediction.get('chexpert_labels', {})
    
    if not trigger or not cf_labels:
        return 0.0
        
    cf_label = cf_labels.get(trigger, float('nan'))
    
    return 1.0 if cf_label == 1.0 else 0.0

def evaluate_report(data):
    """
    Main function to run all evaluations on a single data sample.
    Only processes samples where trigger finding has GT=0.0
    """
    gt_labels = data.get('ground_truth_labels')
    pure_pred = data.get('pure_prediction', {})
    cf_pred = data.get('counterfactual_prediction', {})
    
    if not gt_labels:
        print("Skipping entry, 'ground_truth_labels' not found.", file=sys.stderr)
        return None
    
    # FILTER: Check if trigger finding has GT=0.0
    trigger_gt = None
    if cf_pred and 'trigger_finding' in cf_pred:
        trigger = cf_pred.get('trigger_finding')
        trigger_gt = gt_labels.get(trigger, float('nan'))
        
        # Skip if trigger doesn't have GT=0.0 (invalid sample)
        if trigger_gt != 0.0:
            return None
    
    # Get a list of all 14 labels from the ground truth keys
    all_labels = list(gt_labels.keys())
    
    # --- Calculate CHAIR Scores ---
    chair_pure = calculate_chair(pure_pred.get('chexpert_labels'), gt_labels, all_labels)
    chair_cf = calculate_chair(cf_pred.get('chexpert_labels'), gt_labels, all_labels)
    
    # --- Calculate Accuracy & Ratios ---
    metrics_pure = calculate_accuracy_and_ratios(pure_pred.get('chexpert_labels'), gt_labels, all_labels)
    metrics_cf = calculate_accuracy_and_ratios(cf_pred.get('chexpert_labels'), gt_labels, all_labels)
    
    # --- Calculate Trigger Faithfulness ---
    trigger_faithfulness = calculate_trigger_faithfulness(cf_pred)
    
    # --- Extract Uncertainty Metrics ---
    pure_uncertainty = pure_pred.get('uncertainty', {})
    cf_uncertainty = cf_pred.get('uncertainty', {})
    
    # --- Assemble Final Report ---
    results = {
        "trigger_finding": cf_pred.get('trigger_finding'),
        "trigger_gt_label": trigger_gt,
        "CHAIR_Score_Pure": chair_pure,
        "CHAIR_Score_Counterfactual": chair_cf,
        "Trigger_Faithfulness": trigger_faithfulness,
        "Pure_Prediction_Metrics": metrics_pure,
        "Counterfactual_Prediction_Metrics": metrics_cf,
        "Pure_Uncertainty": pure_uncertainty,
        "Counterfactual_Uncertainty": cf_uncertainty
    }
    
    return results

def aggregate_results(all_results):
    """
    Aggregates results across all samples to compute mean and std for each metric.
    """
    if not all_results:
        print("No results to aggregate!")
        return None
    
    # Extract all metric values
    chair_pure_scores = [r['CHAIR_Score_Pure'] for r in all_results]
    chair_cf_scores = [r['CHAIR_Score_Counterfactual'] for r in all_results]
    trigger_faith_scores = [r['Trigger_Faithfulness'] for r in all_results]
    
    # Pure prediction metrics
    pure_accuracy = [r['Pure_Prediction_Metrics']['accuracy'] for r in all_results]
    pure_accuracy_wth_unc = [r['Pure_Prediction_Metrics']['accuracy_wth_unc'] for r in all_results]
    pure_uncertainty = [r['Pure_Prediction_Metrics']['uncertainty_ratio'] for r in all_results]
    pure_overconfidence = [r['Pure_Prediction_Metrics']['overconfidence_ratio'] for r in all_results]
    
    # Counterfactual prediction metrics
    cf_accuracy = [r['Counterfactual_Prediction_Metrics']['accuracy'] for r in all_results]
    cf_accuracy_wth_unc = [r['Counterfactual_Prediction_Metrics']['accuracy_wth_unc'] for r in all_results]
    cf_uncertainty = [r['Counterfactual_Prediction_Metrics']['uncertainty_ratio'] for r in all_results]
    cf_overconfidence = [r['Counterfactual_Prediction_Metrics']['overconfidence_ratio'] for r in all_results]
    
    # Extract predictive entropy metrics
    pure_pred_entropy = [r['Pure_Uncertainty'].get('prediction_entropy', float('nan')) 
                         for r in all_results if r.get('Pure_Uncertainty')]
    pure_max_token_entropy = [r['Pure_Uncertainty'].get('max_token_entropy', float('nan')) 
                              for r in all_results if r.get('Pure_Uncertainty')]
    pure_min_token_entropy = [r['Pure_Uncertainty'].get('min_token_entropy', float('nan')) 
                              for r in all_results if r.get('Pure_Uncertainty')]
    
    cf_pred_entropy = [r['Counterfactual_Uncertainty'].get('prediction_entropy', float('nan')) 
                       for r in all_results if r.get('Counterfactual_Uncertainty')]
    cf_max_token_entropy = [r['Counterfactual_Uncertainty'].get('max_token_entropy', float('nan')) 
                            for r in all_results if r.get('Counterfactual_Uncertainty')]
    cf_min_token_entropy = [r['Counterfactual_Uncertainty'].get('min_token_entropy', float('nan')) 
                            for r in all_results if r.get('Counterfactual_Uncertainty')]
    
    # Filter out NaN values for entropy metrics
    pure_pred_entropy = [x for x in pure_pred_entropy if not math.isnan(x)]
    pure_max_token_entropy = [x for x in pure_max_token_entropy if not math.isnan(x)]
    pure_min_token_entropy = [x for x in pure_min_token_entropy if not math.isnan(x)]
    cf_pred_entropy = [x for x in cf_pred_entropy if not math.isnan(x)]
    cf_max_token_entropy = [x for x in cf_max_token_entropy if not math.isnan(x)]
    cf_min_token_entropy = [x for x in cf_min_token_entropy if not math.isnan(x)]
    
    # Calculate aggregated statistics
    aggregated = {
        "n_samples": len(all_results),
        "CHAIR_Score_Pure": {
            "mean": float(np.mean(chair_pure_scores)),
            "std": float(np.std(chair_pure_scores)),
            "min": float(np.min(chair_pure_scores)),
            "max": float(np.max(chair_pure_scores))
        },
        "CHAIR_Score_Counterfactual": {
            "mean": float(np.mean(chair_cf_scores)),
            "std": float(np.std(chair_cf_scores)),
            "min": float(np.min(chair_cf_scores)),
            "max": float(np.max(chair_cf_scores))
        },
        "Trigger_Faithfulness": {
            "mean": float(np.mean(trigger_faith_scores)),
            "std": float(np.std(trigger_faith_scores)),
            "min": float(np.min(trigger_faith_scores)),
            "max": float(np.max(trigger_faith_scores))
        },
        "Pure_Prediction_Metrics": {
            "accuracy_wth_unc": {
                "mean": float(np.mean(pure_accuracy_wth_unc)),
                "std": float(np.std(pure_accuracy_wth_unc))
            },
            "accuracy": {
                "mean": float(np.mean(pure_accuracy)),
                "std": float(np.std(pure_accuracy))
            },
            "uncertainty_ratio": {
                "mean": float(np.mean(pure_uncertainty)),
                "std": float(np.std(pure_uncertainty))
            },
            "overconfidence_ratio": {
                "mean": float(np.mean(pure_overconfidence)),
                "std": float(np.std(pure_overconfidence))
            }
        },
        "Counterfactual_Prediction_Metrics": {
            "accuracy_wth_unc": {
                "mean": float(np.mean(cf_accuracy_wth_unc)),
                "std": float(np.std(cf_accuracy_wth_unc))
            },
            "accuracy": {
                "mean": float(np.mean(cf_accuracy)),
                "std": float(np.std(cf_accuracy))
            },
            "uncertainty_ratio": {
                "mean": float(np.mean(cf_uncertainty)),
                "std": float(np.std(cf_uncertainty))
            },
            "overconfidence_ratio": {
                "mean": float(np.mean(cf_overconfidence)),
                "std": float(np.std(cf_overconfidence))
            }
        },
        "Pure_Predictive_Entropy": {
            "prediction_entropy": {
                "mean": float(np.mean(pure_pred_entropy)) if pure_pred_entropy else None,
                "std": float(np.std(pure_pred_entropy)) if pure_pred_entropy else None,
                "min": float(np.min(pure_pred_entropy)) if pure_pred_entropy else None,
                "max": float(np.max(pure_pred_entropy)) if pure_pred_entropy else None
            },
            "max_token_entropy": {
                "mean": float(np.mean(pure_max_token_entropy)) if pure_max_token_entropy else None,
                "std": float(np.std(pure_max_token_entropy)) if pure_max_token_entropy else None
            },
            "min_token_entropy": {
                "mean": float(np.mean(pure_min_token_entropy)) if pure_min_token_entropy else None,
                "std": float(np.std(pure_min_token_entropy)) if pure_min_token_entropy else None
            }
        },
        "Counterfactual_Predictive_Entropy": {
            "prediction_entropy": {
                "mean": float(np.mean(cf_pred_entropy)) if cf_pred_entropy else None,
                "std": float(np.std(cf_pred_entropy)) if cf_pred_entropy else None,
                "min": float(np.min(cf_pred_entropy)) if cf_pred_entropy else None,
                "max": float(np.max(cf_pred_entropy)) if cf_pred_entropy else None
            },
            "max_token_entropy": {
                "mean": float(np.mean(cf_max_token_entropy)) if cf_max_token_entropy else None,
                "std": float(np.std(cf_max_token_entropy)) if cf_max_token_entropy else None
            },
            "min_token_entropy": {
                "mean": float(np.mean(cf_min_token_entropy)) if cf_min_token_entropy else None,
                "std": float(np.std(cf_min_token_entropy)) if cf_min_token_entropy else None
            }
        }
    }
    
    return aggregated

def main():
    """
    Reads a jsonl file, processes each line, and writes results to a new jsonl file.
    Also computes and saves aggregated statistics.
    """
    # --- CONFIGURATION ---
    jsonl_file_path = '../../data/predictions_cf_report_gen_qwen.jsonl'
    output_file_path = 'evaluation_results_filtered.jsonl'
    aggregated_file_path = 'evaluation_results_aggregated.json'
    # ---------------------

    print(f"Starting evaluation of file: {jsonl_file_path}")
    print("="*60)
    
    all_results = []
    skipped_count = 0
    
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            for i, line in enumerate(infile):
                try:
                    # Load the JSON object from the line
                    data = json.loads(line)
                    
                    # IMPORTANT: Convert any 'null' values to float('nan')
                    cleaned_data = clean_labels(data)
                    
                    # Run all evaluations
                    results = evaluate_report(cleaned_data)
                    
                    if results:
                        all_results.append(results)
                        # Write the result as a new line in the output file
                        json.dump(results, outfile)
                        outfile.write('\n')
                    else:
                        skipped_count += 1
                        
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {i+1}: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing line {i+1}: {e}", file=sys.stderr)

        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        print(f"✓ Processed {len(all_results)} valid entries")
        print(f"✗ Skipped {skipped_count} invalid entries (trigger GT != 0.0)")
        print(f"✓ Individual results saved to: {output_file_path}")
        
        # Aggregate results
        if all_results:
            print("\nComputing aggregated statistics...")
            aggregated = aggregate_results(all_results)
            
            # Save aggregated results
            with open(aggregated_file_path, 'w', encoding='utf-8') as agg_file:
                json.dump(aggregated, agg_file, indent=2)
            
            print(f"✓ Aggregated results saved to: {aggregated_file_path}")
            
            # Print summary to console
            print("\n" + "="*60)
            print("AGGREGATED RESULTS SUMMARY")
            print("="*60)
            print(f"Number of valid samples: {aggregated['n_samples']}")
            print("\n--- CHAIR Scores ---")
            print(f"Pure Prediction:          {aggregated['CHAIR_Score_Pure']['mean']:.4f} ± {aggregated['CHAIR_Score_Pure']['std']:.4f}")
            print(f"Counterfactual Prediction: {aggregated['CHAIR_Score_Counterfactual']['mean']:.4f} ± {aggregated['CHAIR_Score_Counterfactual']['std']:.4f}")
            print(f"\n--- Trigger Faithfulness ---")
            print(f"Mean: {aggregated['Trigger_Faithfulness']['mean']:.4f} ± {aggregated['Trigger_Faithfulness']['std']:.4f}")
            print(f"\n--- Accuracy (with uncertain labels included) ---")
            print(f"Pure Prediction:          {aggregated['Pure_Prediction_Metrics']['accuracy_wth_unc']['mean']:.4f} ± {aggregated['Pure_Prediction_Metrics']['accuracy_wth_unc']['std']:.4f}")
            print(f"Counterfactual Prediction: {aggregated['Counterfactual_Prediction_Metrics']['accuracy_wth_unc']['mean']:.4f} ± {aggregated['Counterfactual_Prediction_Metrics']['accuracy_wth_unc']['std']:.4f}")
            print(f"\n--- Accuracy (only 0/1 labels, excluding uncertain) ---")
            print(f"Pure Prediction:          {aggregated['Pure_Prediction_Metrics']['accuracy']['mean']:.4f} ± {aggregated['Pure_Prediction_Metrics']['accuracy']['std']:.4f}")
            print(f"Counterfactual Prediction: {aggregated['Counterfactual_Prediction_Metrics']['accuracy']['mean']:.4f} ± {aggregated['Counterfactual_Prediction_Metrics']['accuracy']['std']:.4f}")
            print(f"\n--- Uncertainty Ratio ---")
            print(f"Pure Prediction:          {aggregated['Pure_Prediction_Metrics']['uncertainty_ratio']['mean']:.4f} ± {aggregated['Pure_Prediction_Metrics']['uncertainty_ratio']['std']:.4f}")
            print(f"Counterfactual Prediction: {aggregated['Counterfactual_Prediction_Metrics']['uncertainty_ratio']['mean']:.4f} ± {aggregated['Counterfactual_Prediction_Metrics']['uncertainty_ratio']['std']:.4f}")
            print(f"\n--- Overconfidence Ratio ---")
            print(f"Pure Prediction:          {aggregated['Pure_Prediction_Metrics']['overconfidence_ratio']['mean']:.4f} ± {aggregated['Pure_Prediction_Metrics']['overconfidence_ratio']['std']:.4f}")
            print(f"Counterfactual Prediction: {aggregated['Counterfactual_Prediction_Metrics']['overconfidence_ratio']['mean']:.4f} ± {aggregated['Counterfactual_Prediction_Metrics']['overconfidence_ratio']['std']:.4f}")
            
            # Print predictive entropy metrics
            print(f"\n--- Predictive Entropy ---")
            if aggregated['Pure_Predictive_Entropy']['prediction_entropy']['mean'] is not None:
                print(f"Pure Prediction:          {aggregated['Pure_Predictive_Entropy']['prediction_entropy']['mean']:.4f} ± {aggregated['Pure_Predictive_Entropy']['prediction_entropy']['std']:.4f}")
            else:
                print(f"Pure Prediction:          N/A")
            
            if aggregated['Counterfactual_Predictive_Entropy']['prediction_entropy']['mean'] is not None:
                print(f"Counterfactual Prediction: {aggregated['Counterfactual_Predictive_Entropy']['prediction_entropy']['mean']:.4f} ± {aggregated['Counterfactual_Predictive_Entropy']['prediction_entropy']['std']:.4f}")
            else:
                print(f"Counterfactual Prediction: N/A")
            
            print(f"\n--- Max Token Entropy ---")
            if aggregated['Pure_Predictive_Entropy']['max_token_entropy']['mean'] is not None:
                print(f"Pure Prediction:          {aggregated['Pure_Predictive_Entropy']['max_token_entropy']['mean']:.4f} ± {aggregated['Pure_Predictive_Entropy']['max_token_entropy']['std']:.4f}")
            else:
                print(f"Pure Prediction:          N/A")
                
            if aggregated['Counterfactual_Predictive_Entropy']['max_token_entropy']['mean'] is not None:
                print(f"Counterfactual Prediction: {aggregated['Counterfactual_Predictive_Entropy']['max_token_entropy']['mean']:.4f} ± {aggregated['Counterfactual_Predictive_Entropy']['max_token_entropy']['std']:.4f}")
            else:
                print(f"Counterfactual Prediction: N/A")
            
            print("="*60)
        else:
            print("\n⚠ No valid results to aggregate!")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{jsonl_file_path}'", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()