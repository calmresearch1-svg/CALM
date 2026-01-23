#!/usr/bin/env python3
"""
Counterfactual Fairness Metrics Calculator for FairVLM Dataset.
Calculates fairness metrics from the output of run_qwen_fairvlm_npz.py.
Handles two separate JSONL files (baseline and counterfactual).
"""

import argparse
import json
import re
from typing import Any, Dict, List
from collections import defaultdict


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


def extract_yes_no_answer(model_output: str) -> str:
    """
    Extract yes/no answer from model output.
    Tries multiple strategies in order:
    1. JSON parsing
    2. Quoted strings: \"no\", 'no', \"yes\", 'yes'
    3. Word boundaries: yes/no with space before and non-letter after
    Returns "X" if not detected or ambiguous.
    
    Args:
        model_output: Raw model output string
        
    Returns:
        The predicted answer ("yes", "no", or "X" if extraction fails)
    """
    # TRY 1: JSON PARSING (The strict way)
    try:
        json_start = model_output.find('{')
        json_end = model_output.rfind('}')
        
        if json_start == -1 or json_end == -1 or json_end < json_start:
            raise ValueError("Could not find JSON object boundaries.")
        
        json_string = model_output[json_start : json_end + 1]
        data = json.loads(json_string)
        predicted_answer = data.get("answer")
        
        if predicted_answer is None:
            raise ValueError("JSON object missing 'answer' key.")
        
        # Check if answer is "yes" or "no"
        if predicted_answer.lower() in ["yes", "no"]:
            return predicted_answer.lower()
        else:
            return "X"
        
    except (json.JSONDecodeError, ValueError):
        pass  # Continue to next strategy
    
    # TRY 2: Look for quoted strings
    # Check for \"no\"
    if '\"no\"' in model_output:
        return "no"
    
    # Check for 'no'
    if "'no'" in model_output:
        return "no"
    
    # Check for \"yes\"
    if '\"yes\"' in model_output:
        return "yes"
    
    # Check for 'yes'
    if "'yes'" in model_output:
        return "yes"
    
    # TRY 3: Word boundary search
    # Pattern: (space or start) + yes/no + (non-letter or end)
    yes_pattern = r'(?:^|\s)(yes)(?:[^a-zA-Z]|$)'
    no_pattern = r'(?:^|\s)(no)(?:[^a-zA-Z]|$)'
    
    yes_match = re.search(yes_pattern, model_output, re.IGNORECASE)
    no_match = re.search(no_pattern, model_output, re.IGNORECASE)
    
    # If both found, ambiguous
    if yes_match and no_match:
        return "X"
    
    # If only yes found
    if yes_match:
        return "yes"
    
    # If only no found
    if no_match:
        return "no"
    
    # If neither found
    return "X"


def calculate_confusion_matrix_metrics(group_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate confusion matrix metrics for a group.
    
    Args:
        group_data: List of samples in the group
        
    Returns:
        Dictionary with TP, FN, FP, TN, N, Acc, TPR, FPR, TNR, FNR
    """
    tp = 0  # True Positive: predicted yes, actual yes
    fn = 0  # False Negative: predicted no, actual yes  
    fp = 0  # False Positive: predicted yes, actual no
    tn = 0  # True Negative: predicted no, actual no
    
    for sample in group_data:
        predicted = sample.get("predicted_answer", "X")
        actual = sample.get("correct_answer", "").lower()
        
        if predicted == "X":  # Skip invalid predictions
            continue
            
        if actual == "yes":
            if predicted == "yes":
                tp += 1
            elif predicted == "no":
                fn += 1
        elif actual == "no":
            if predicted == "yes":
                fp += 1
            elif predicted == "no":
                tn += 1
    
    # Calculate metrics
    n = tp + fn + fp + tn
    acc = (tp + tn) / n if n > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity/Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
    
    return {
        "TP": tp,
        "FN": fn, 
        "FP": fp,
        "TN": tn,
        "N": n,
        "Acc": acc,
        "TPR": tpr,
        "FPR": fpr,
        "TNR": tnr,
        "FNR": fnr
    }


def analyze_demographic_group(data: List[Dict[str, Any]], condition: str, group_attr: str, group_name: str) -> Dict[str, Any]:
    """
    Analyze a demographic group for fairness metrics.
    
    Args:
        data: All data
        condition: Condition to filter by (e.g., "baseline_gender", "counterfactual_age")
        group_attr: Attribute to group by (e.g., "ground_truth_gender", "ground_truth_age")
        group_name: Name for display (e.g., "Gender", "Age")
    
    Returns:
        Dictionary with group metrics and fairness gaps
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {condition.upper().replace('_', ' ')} - {group_name}")
    print(f"{'='*60}")
    
    # Filter for condition
    condition_data = [row for row in data if row.get("condition") == condition]
    
    if not condition_data:
        print(f"No {condition} data found.")
        return {}
    
    # Count predicted answers
    yes_count = sum(1 for row in condition_data if row.get("predicted_answer") == "yes")
    no_count = sum(1 for row in condition_data if row.get("predicted_answer") == "no")
    undetected_count = sum(1 for row in condition_data if row.get("predicted_answer") == "X")
    
    print(f"Total samples: {len(condition_data)}")
    print(f"Predicted 'yes': {yes_count}")
    print(f"Predicted 'no': {no_count}")
    print(f"Undetected/Invalid: {undetected_count}")
    
    # Group by demographic attribute
    groups = defaultdict(list)
    for row in condition_data:
        attr_value = row.get(group_attr)
        if attr_value is not None:
            groups[attr_value].append(row)
    
    if not groups:
        print(f"No {group_name} groups found.")
        return {}
    
    # Calculate metrics for each group
    group_metrics = {}
    for group_value in sorted(groups.keys()):
        group_data = groups[group_value]
        metrics = calculate_confusion_matrix_metrics(group_data)
        group_metrics[group_value] = metrics
        
        print(f"{group_name} {group_value} (N={metrics['N']}):")
        print(f"  Accuracy: {metrics['Acc']:.4f}")
        print(f"  TPR: {metrics['TPR']:.4f}")
        print(f"  FPR: {metrics['FPR']:.4f}")
        print(f"  TNR: {metrics['TNR']:.4f}")
    
    # Calculate fairness gaps
    accuracies = [metrics['Acc'] for metrics in group_metrics.values()]
    tprs = [metrics['TPR'] for metrics in group_metrics.values()]
    fprs = [metrics['FPR'] for metrics in group_metrics.values()]
    
    # Demographic Accuracy Difference
    dad = max(accuracies) - min(accuracies)
    
    # Max-Min Fairness (worst accuracy group)
    max_min_fairness = min(accuracies)
    
    # Demographic Parity Difference (DPD)
    positive_pred_rates = [(metrics['TP'] + metrics['FP']) / metrics['N'] for metrics in group_metrics.values() if metrics['N'] > 0]
    dpd = max(positive_pred_rates) - min(positive_pred_rates) if positive_pred_rates else 0.0
    
    # Equal Opportunity Difference (EOD)
    eod = max(tprs) - min(tprs)
    
    # Difference in Equalized Odds (DEOdds)
    max_tpr_gap = max(tprs) - min(tprs)
    max_fpr_gap = max(fprs) - min(fprs)
    deodds = 0.5 * (max_tpr_gap + max_fpr_gap)
    
    print(f"\nFairness Gaps:")
    print(f"  Demographic Accuracy Difference (DAD): {dad:.4f}")
    print(f"  Max-Min Fairness: {max_min_fairness:.4f}")
    print(f"  Demographic Parity Difference (DPD): {dpd:.4f}")
    print(f"  Equal Opportunity Difference (EOD): {eod:.4f}")
    print(f"  Difference in Equalized Odds (DEOdds): {deodds:.4f}")
    
    return {
        "group_metrics": group_metrics,
        "dad": dad,
        "max_min_fairness": max_min_fairness,
        "dpd": dpd,
        "eod": eod,
        "deodds": deodds
    }


def analyze_age_groups(data: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    """
    Analyze age groups (elderly vs non-elderly) for fairness metrics.
    
    Args:
        data: All data
        condition: Condition to filter by
    
    Returns:
        Dictionary with group metrics and fairness gaps
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {condition.upper().replace('_', ' ')} - AGE")
    print(f"{'='*60}")
    
    # Filter for condition
    condition_data = [row for row in data if row.get("condition") == condition]
    
    if not condition_data:
        print(f"No {condition} data found.")
        return {}
    
    # Count predicted answers
    yes_count = sum(1 for row in condition_data if row.get("predicted_answer") == "yes")
    no_count = sum(1 for row in condition_data if row.get("predicted_answer") == "no")
    undetected_count = sum(1 for row in condition_data if row.get("predicted_answer") == "X")
    
    print(f"Total samples: {len(condition_data)}")
    print(f"Predicted 'yes': {yes_count}")
    print(f"Predicted 'no': {no_count}")
    print(f"Undetected/Invalid: {undetected_count}")
    
    # Group by age category (elderly vs non-elderly)
    elderly_data = [row for row in condition_data if row.get("ground_truth_age", 0) >= 65]
    non_elderly_data = [row for row in condition_data if row.get("ground_truth_age", 0) < 65]
    
    # Calculate metrics for each group
    elderly_metrics = calculate_confusion_matrix_metrics(elderly_data)
    non_elderly_metrics = calculate_confusion_matrix_metrics(non_elderly_data)
    
    print(f"Elderly Group (N={elderly_metrics['N']}):")
    print(f"  Accuracy: {elderly_metrics['Acc']:.4f}")
    print(f"  TPR: {elderly_metrics['TPR']:.4f}")
    print(f"  FPR: {elderly_metrics['FPR']:.4f}")
    print(f"  TNR: {elderly_metrics['TNR']:.4f}")
    
    print(f"\nNon-Elderly Group (N={non_elderly_metrics['N']}):")
    print(f"  Accuracy: {non_elderly_metrics['Acc']:.4f}")
    print(f"  TPR: {non_elderly_metrics['TPR']:.4f}")
    print(f"  FPR: {non_elderly_metrics['FPR']:.4f}")
    print(f"  TNR: {non_elderly_metrics['TNR']:.4f}")
    
    # Calculate fairness gaps
    accuracies = [elderly_metrics['Acc'], non_elderly_metrics['Acc']]
    tprs = [elderly_metrics['TPR'], non_elderly_metrics['TPR']]
    fprs = [elderly_metrics['FPR'], non_elderly_metrics['FPR']]
    
    # Demographic Accuracy Difference
    dad = max(accuracies) - min(accuracies)
    
    # Max-Min Fairness (worst accuracy group)
    max_min_fairness = min(accuracies)
    
    # Demographic Parity Difference (DPD)
    elderly_ppr = (elderly_metrics['TP'] + elderly_metrics['FP']) / elderly_metrics['N'] if elderly_metrics['N'] > 0 else 0
    non_elderly_ppr = (non_elderly_metrics['TP'] + non_elderly_metrics['FP']) / non_elderly_metrics['N'] if non_elderly_metrics['N'] > 0 else 0
    dpd = abs(elderly_ppr - non_elderly_ppr)
    
    # Equal Opportunity Difference (EOD)
    eod = max(tprs) - min(tprs)
    
    # Difference in Equalized Odds (DEOdds)
    max_tpr_gap = max(tprs) - min(tprs)
    max_fpr_gap = max(fprs) - min(fprs)
    deodds = 0.5 * (max_tpr_gap + max_fpr_gap)
    
    print(f"\nFairness Gaps:")
    print(f"  Demographic Accuracy Difference (DAD): {dad:.4f}")
    print(f"  Max-Min Fairness: {max_min_fairness:.4f}")
    print(f"  Demographic Parity Difference (DPD): {dpd:.4f}")
    print(f"  Equal Opportunity Difference (EOD): {eod:.4f}")
    print(f"  Difference in Equalized Odds (DEOdds): {deodds:.4f}")
    
    return {
        "elderly_metrics": elderly_metrics,
        "non_elderly_metrics": non_elderly_metrics,
        "dad": dad,
        "max_min_fairness": max_min_fairness,
        "dpd": dpd,
        "eod": eod,
        "deodds": deodds
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate Counterfactual Fairness Metrics for FairVLM Dataset")
    parser.add_argument(
        "--baseline_jsonl",
        type=str,
        required=True,
        help="Path to baseline JSONL file from run_qwen_fairvlm_npz.py"
    )
    parser.add_argument(
        "--counterfactual_jsonl",
        type=str,
        required=True,
        help="Path to counterfactual JSONL file from run_qwen_fairvlm_npz.py"
    )
    parser.add_argument(
        "--output_baseline",
        type=str,
        default=None,
        help="Optional: Path to save processed baseline data with predicted answers"
    )
    parser.add_argument(
        "--output_counterfactual",
        type=str,
        default=None,
        help="Optional: Path to save processed counterfactual data with predicted answers"
    )
    args = parser.parse_args()
    
    # Load data from both files
    print(f"Loading baseline data from: {args.baseline_jsonl}")
    baseline_data = load_jsonl(args.baseline_jsonl)
    print(f"Loaded {len(baseline_data)} baseline samples")
    
    print(f"Loading counterfactual data from: {args.counterfactual_jsonl}")
    counterfactual_data = load_jsonl(args.counterfactual_jsonl)
    print(f"Loaded {len(counterfactual_data)} counterfactual samples")
    
    # Merge both datasets
    all_data = baseline_data + counterfactual_data
    print(f"Total samples: {len(all_data)}")
    
    # Process each row to add predicted_answer
    print("\nProcessing model outputs...")
    processed_data = []
    
    for row in all_data:
        # Extract predicted answer
        predicted_answer = extract_yes_no_answer(row.get("model_output", ""))
        
        # Add new field
        processed_row = row.copy()
        processed_row["predicted_answer"] = predicted_answer
        
        processed_data.append(processed_row)
    
    # Save processed data if output paths provided
    if args.output_baseline:
        print(f"Saving processed baseline data to: {args.output_baseline}")
        baseline_processed = [row for row in processed_data if row.get("condition", "").startswith("baseline")]
        with open(args.output_baseline, "w", encoding="utf-8") as f:
            for row in baseline_processed:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    if args.output_counterfactual:
        print(f"Saving processed counterfactual data to: {args.output_counterfactual}")
        cf_processed = [row for row in processed_data if row.get("condition", "").startswith("counterfactual")]
        with open(args.output_counterfactual, "w", encoding="utf-8") as f:
            for row in cf_processed:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    # Run all analyses
    print("\n" + "="*80)
    print("COUNTERFACTUAL FAIRNESS METRICS ANALYSIS - FAIRVLM DATASET")
    print("="*80)
    
    # Baseline analyses
    print("\n" + "="*80)
    print("BASELINE ANALYSES")
    print("="*80)
    
    analyze_demographic_group(processed_data, "baseline_gender", "ground_truth_gender", "Gender")
    analyze_age_groups(processed_data, "baseline_age")
    analyze_demographic_group(processed_data, "baseline_race", "ground_truth_race", "Race")
    
    # Counterfactual analyses
    print("\n" + "="*80)
    print("COUNTERFACTUAL ANALYSES")
    print("="*80)
    
    analyze_demographic_group(processed_data, "counterfactual_gender", "ground_truth_gender", "Gender")
    analyze_age_groups(processed_data, "counterfactual_age")
    analyze_demographic_group(processed_data, "counterfactual_race", "ground_truth_race", "Race")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

