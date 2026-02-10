#!/usr/bin/env python3
"""
Counterfactual Fairness Metrics Calculator for HAM10000 Dataset.
Calculates fairness metrics from the output of run_qwen_cf_experiment.py / run_smolvlm_cf_experiment.py.
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


def extract_multichoice_answer(model_output: str) -> str:
    """
    Extract answer from model output using the exact 2-step logic from extract_answer.py.
    
    Args:
        model_output: Raw model output string
        
    Returns:
        The predicted option letter (A, B, C, or D) or "X" if extraction fails
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
        
        return predicted_answer
        
    except (json.JSONDecodeError, ValueError):
        # TRY 2: REGEX FALLBACK (The robust way)
        try:
            answer_pattern = re.compile(r'"answer"\s*:\s*"([A-Z])"')
            search = answer_pattern.search(model_output)
            if search:
                return search.group(1)  # Get the captured letter
            else:
                raise ValueError("Regex did not find answer pattern.")
        except Exception:
            # TRY 3: Look for plain-text pattern like "Answer: X"
            try:
                plain_pattern = re.compile(r'Answer:\s*([A-Z])')
                search = plain_pattern.search(model_output)
                if search:
                    return search.group(1)
            except Exception:
                pass

            # TRY 4: Look for plain-text pattern like "answer is X" / "answer is option X"
            # (case-insensitive; accept optional "option")
            try:
                answer_is_pattern = re.compile(r'answer\s+is\s+(?:option\s+)?([A-Z])', re.IGNORECASE)
                search = answer_is_pattern.search(model_output)
                if search:
                    return search.group(1).upper()
            except Exception:
                pass

            # TRY 5: Handle truncated JSON like: "answer": "X   (missing closing quote)
            try:
                truncated_json_pattern = re.compile(r'"answer"\s*:\s*"([A-Z])', re.IGNORECASE)
                search = truncated_json_pattern.search(model_output)
                if search:
                    return search.group(1).upper()
            except Exception:
                pass

            # TRY 6: Look for bracketed pattern like "answer is [X"
            try:
                bracket_pattern = re.compile(r'answer\s+is\s*\[\s*([A-Z])\s*', re.IGNORECASE)
                search = bracket_pattern.search(model_output)
                if search:
                    return search.group(1).upper()
            except Exception:
                pass

            # TRY 7: Look for pattern like "correct answer is:\n    [X"
            try:
                correct_answer_pattern = re.compile(
                    r'correct\s+answer\s+is:\s*\n?\s*\[\s*([A-Z])\s*',
                    re.IGNORECASE
                )
                search = correct_answer_pattern.search(model_output)
                if search:
                    return search.group(1).upper()
            except Exception:
                pass

            # ALL METHODS FAILED
            return "X"


def calculate_group_metrics(group_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate metrics for a group of multiple choice samples.
    
    Args:
        group_data: List of samples in the group
        
    Returns:
        Dictionary with N, Acc, Clear_Rate, Correct, Clear
    """
    if not group_data:
        return {"N": 0, "Acc": 0.0, "Clear_Rate": 0.0, "Correct": 0, "Clear": 0}
    
    total_samples = len(group_data)
    correct_predictions = sum(1 for sample in group_data if sample.get("is_correct", False))
    clear_predictions = sum(1 for sample in group_data if sample.get("predicted_answer", "X") != "X")
    
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    clear_rate = clear_predictions / total_samples if total_samples > 0 else 0.0
    
    return {
        "N": total_samples,
        "Acc": accuracy,
        "Clear_Rate": clear_rate,
        "Correct": correct_predictions,
        "Clear": clear_predictions
    }


def analyze_demographic_group(data: List[Dict[str, Any]], condition: str, group_attr: str, group_name: str) -> Dict[str, Any]:
    """
    Analyze a demographic group for fairness metrics.
    
    Args:
        data: All data
        condition: Condition to filter by (e.g., "baseline_gender", "counterfactual_age")
        group_attr: Attribute to group by (e.g., "ground_truth_gender")
        group_name: Name for display (e.g., "Gender")
    
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
    valid_answers = ["A", "B", "C", "D"]
    answer_counts = {ans: sum(1 for row in condition_data if row.get("predicted_answer") == ans) for ans in valid_answers}
    undetected_count = sum(1 for row in condition_data if row.get("predicted_answer") == "X")
    
    print(f"Total samples: {len(condition_data)}")
    for ans in valid_answers:
        print(f"Predicted '{ans}': {answer_counts[ans]}")
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
        metrics = calculate_group_metrics(group_data)
        group_metrics[group_value] = metrics
        
        print(f"\n{group_name} {group_value} (N={metrics['N']}):")
        print(f"  Accuracy: {metrics['Acc']:.4f}")
        print(f"  Clear Rate: {metrics['Clear_Rate']:.4f}")
        print(f"  Correct: {metrics['Correct']} / Clear: {metrics['Clear']} / Total: {metrics['N']}")
    
    # Calculate fairness gaps
    accuracies = [metrics['Acc'] for metrics in group_metrics.values()]
    clear_rates = [metrics['Clear_Rate'] for metrics in group_metrics.values()]
    
    # Demographic Accuracy Difference
    dad = max(accuracies) - min(accuracies)
    
    # Max-Min Fairness (worst accuracy group)
    max_min_fairness = min(accuracies)
    
    # Clear Rate Difference
    clear_rate_diff = max(clear_rates) - min(clear_rates)
    
    print(f"\nFairness Gaps:")
    print(f"  Demographic Accuracy Difference (DAD): {dad:.4f}")
    print(f"  Max-Min Fairness: {max_min_fairness:.4f}")
    print(f"  Clear Rate Difference: {clear_rate_diff:.4f}")
    
    return {
        "group_metrics": group_metrics,
        "dad": dad,
        "max_min_fairness": max_min_fairness,
        "clear_rate_diff": clear_rate_diff
    }


def analyze_age_groups(data: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    """
    Analyze age groups (middle-aged vs non-middle-aged) for fairness metrics.
    
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
    valid_answers = ["A", "B", "C", "D"]
    answer_counts = {ans: sum(1 for row in condition_data if row.get("predicted_answer") == ans) for ans in valid_answers}
    undetected_count = sum(1 for row in condition_data if row.get("predicted_answer") == "X")
    
    print(f"Total samples: {len(condition_data)}")
    for ans in valid_answers:
        print(f"Predicted '{ans}': {answer_counts[ans]}")
    print(f"Undetected/Invalid: {undetected_count}")
    
    # Group by age category (middle-aged 40-65 vs non-middle-aged <40 or >65)
    middle_aged_data = [row for row in condition_data if 40 <= row.get("ground_truth_age", 0) <= 65]
    non_middle_aged_data = [row for row in condition_data if row.get("ground_truth_age", 0) < 40 or row.get("ground_truth_age", 0) > 65]
    
    # Calculate metrics for each group
    middle_aged_metrics = calculate_group_metrics(middle_aged_data)
    non_middle_aged_metrics = calculate_group_metrics(non_middle_aged_data)
    
    print(f"\nMiddle-Aged Group (40-65) (N={middle_aged_metrics['N']}):")
    print(f"  Accuracy: {middle_aged_metrics['Acc']:.4f}")
    print(f"  Clear Rate: {middle_aged_metrics['Clear_Rate']:.4f}")
    print(f"  Correct: {middle_aged_metrics['Correct']} / Clear: {middle_aged_metrics['Clear']} / Total: {middle_aged_metrics['N']}")
    
    print(f"\nNon-Middle-Aged Group (<40 or >65) (N={non_middle_aged_metrics['N']}):")
    print(f"  Accuracy: {non_middle_aged_metrics['Acc']:.4f}")
    print(f"  Clear Rate: {non_middle_aged_metrics['Clear_Rate']:.4f}")
    print(f"  Correct: {non_middle_aged_metrics['Correct']} / Clear: {non_middle_aged_metrics['Clear']} / Total: {non_middle_aged_metrics['N']}")
    
    # Calculate fairness gaps
    accuracies = [middle_aged_metrics['Acc'], non_middle_aged_metrics['Acc']]
    clear_rates = [middle_aged_metrics['Clear_Rate'], non_middle_aged_metrics['Clear_Rate']]
    
    # Demographic Accuracy Difference
    dad = max(accuracies) - min(accuracies)
    
    # Max-Min Fairness (worst accuracy group)
    max_min_fairness = min(accuracies)
    
    # Clear Rate Difference
    clear_rate_diff = max(clear_rates) - min(clear_rates)
    
    print(f"\nFairness Gaps:")
    print(f"  Demographic Accuracy Difference (DAD): {dad:.4f}")
    print(f"  Max-Min Fairness: {max_min_fairness:.4f}")
    print(f"  Clear Rate Difference: {clear_rate_diff:.4f}")
    
    return {
        "middle_aged_metrics": middle_aged_metrics,
        "non_middle_aged_metrics": non_middle_aged_metrics,
        "dad": dad,
        "max_min_fairness": max_min_fairness,
        "clear_rate_diff": clear_rate_diff
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate Counterfactual Fairness Metrics for HAM10000 Dataset")
    parser.add_argument(
        "--baseline_jsonl",
        type=str,
        required=True,
        help="Path to baseline JSONL file from run_qwen_cf_experiment.py or run_smolvlm_cf_experiment.py"
    )
    parser.add_argument(
        "--counterfactual_jsonl",
        type=str,
        required=True,
        help="Path to counterfactual JSONL file from run_qwen_cf_experiment.py or run_smolvlm_cf_experiment.py"
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
    parser.add_argument(
        "--output_metrics",
        type=str,
        default=None,
        help="Optional: Path to save calculated metrics to a JSON file"
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
    
    # Process each row to add predicted_answer and is_correct
    print("\nProcessing model outputs...")
    processed_data = []
    
    for row in all_data:
        # Extract predicted answer
        predicted_answer = extract_multichoice_answer(row.get("model_output", ""))
        
        # Calculate is_correct
        correct_answer = row.get("correct_answer", "")
        is_correct = (predicted_answer == correct_answer) and (predicted_answer != "X")
        
        # Add new fields
        processed_row = row.copy()
        processed_row["predicted_answer"] = predicted_answer
        processed_row["is_correct"] = is_correct
        
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
    print("COUNTERFACTUAL FAIRNESS METRICS ANALYSIS - HAM10000 DATASET")
    print("="*80)
    
    # Baseline analyses
    print("\n" + "="*80)
    print("BASELINE ANALYSES")
    print("="*80)
    
    results = {}
    
    results["baseline_gender"] = analyze_demographic_group(processed_data, "baseline_gender", "ground_truth_gender", "Gender")
    results["baseline_age"] = analyze_age_groups(processed_data, "baseline_age")
    
    # Counterfactual analyses
    print("\n" + "="*80)
    print("COUNTERFACTUAL ANALYSES")
    print("="*80)
    
    results["counterfactual_gender"] = analyze_demographic_group(processed_data, "counterfactual_gender", "ground_truth_gender", "Gender")
    results["counterfactual_age"] = analyze_age_groups(processed_data, "counterfactual_age")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    if args.output_metrics:
        print(f"Saving metrics to: {args.output_metrics}")
        import os
        os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
        with open(args.output_metrics, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print("Metrics saved successfully.")

if __name__ == "__main__":
    main()

