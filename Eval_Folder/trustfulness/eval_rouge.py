import json
import sys
import argparse
import re
import numpy as np
from rouge_score import rouge_scorer


def clean_report(text):
    """
    Clean report text by removing all symbols except letters, numbers, spaces, commas, and periods.
    
    Args:
        text: Input text string
    
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return text
    # Keep only alphanumeric characters, spaces, commas, and periods
    cleaned = re.sub(r'[^a-zA-Z0-9\s,.]', '', text)
    # Normalize whitespace (multiple spaces to single space)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def calculate_rouge_l(candidate, reference):
    """
    Calculates ROUGE-L score for a candidate text against a reference text.
    
    Args:
        candidate: Generated text (string)
        reference: Ground truth text (string)
    
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    if not candidate or not reference:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'fmeasure': 0.0
        }
    
    try:
        # Initialize ROUGE scorer for ROUGE-L
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Calculate ROUGE-L scores
        scores = scorer.score(reference, candidate)
        
        return {
            'precision': float(scores['rougeL'].precision),
            'recall': float(scores['rougeL'].recall),
            'fmeasure': float(scores['rougeL'].fmeasure)
        }
    except Exception as e:
        print(f"Error calculating ROUGE-L: {e}", file=sys.stderr)
        return {
            'precision': 0.0,
            'recall': 0.0,
            'fmeasure': 0.0
        }

def evaluate_reports(data):
    """
    Evaluates ROUGE-L scores for baseline and counterfactual reports.
    
    Args:
        data: Dictionary containing the reports
    
    Returns:
        Dictionary with ROUGE-L scores or None if data is invalid
    """
    # Extract ground truth report from 'response' field
    ground_truth = data.get('response')
    
    if not ground_truth:
        print("Warning: No ground truth report found in entry", file=sys.stderr)
        return None
    
    # Extract baseline (pure_prediction_cleaned) and counterfactual (cf_prediction_cleaned) responses
    baseline_response = data.get('pure_prediction_cleaned')
    counterfactual_response = data.get('cf_prediction_cleaned')
    
    if not baseline_response and not counterfactual_response:
        print("Warning: No baseline or counterfactual responses found", file=sys.stderr)
        return None
    
    # Clean all texts before calculating scores
    ground_truth = clean_report(ground_truth)
    baseline_response = clean_report(baseline_response) if baseline_response else None
    counterfactual_response = clean_report(counterfactual_response) if counterfactual_response else None
    
    # Calculate ROUGE-L scores
    results = {}
    
    if baseline_response:
        results['baseline_rouge_l'] = calculate_rouge_l(baseline_response, ground_truth)
    else:
        results['baseline_rouge_l'] = None
    
    if counterfactual_response:
        results['counterfactual_rouge_l'] = calculate_rouge_l(counterfactual_response, ground_truth)
    else:
        results['counterfactual_rouge_l'] = None
    
    return results

def aggregate_results(all_results):
    """
    Aggregates ROUGE-L scores across all samples.
    
    Args:
        all_results: List of result dictionaries
    
    Returns:
        Dictionary with aggregated statistics
    """
    if not all_results:
        print("No results to aggregate!")
        return None
    
    # Extract scores, filtering out None values
    baseline_precision = [r['baseline_rouge_l']['precision'] for r in all_results 
                         if r.get('baseline_rouge_l') is not None]
    baseline_recall = [r['baseline_rouge_l']['recall'] for r in all_results 
                      if r.get('baseline_rouge_l') is not None]
    baseline_fmeasure = [r['baseline_rouge_l']['fmeasure'] for r in all_results 
                        if r.get('baseline_rouge_l') is not None]
    
    counterfactual_precision = [r['counterfactual_rouge_l']['precision'] for r in all_results 
                               if r.get('counterfactual_rouge_l') is not None]
    counterfactual_recall = [r['counterfactual_rouge_l']['recall'] for r in all_results 
                            if r.get('counterfactual_rouge_l') is not None]
    counterfactual_fmeasure = [r['counterfactual_rouge_l']['fmeasure'] for r in all_results 
                              if r.get('counterfactual_rouge_l') is not None]
    
    aggregated = {
        "n_samples": len(all_results),
        "n_baseline_samples": len(baseline_fmeasure),
        "n_counterfactual_samples": len(counterfactual_fmeasure)
    }
    
    if baseline_fmeasure:
        aggregated["baseline"] = {
            "precision": {
                "mean": float(np.mean(baseline_precision)),
                "std": float(np.std(baseline_precision)),
                "min": float(np.min(baseline_precision)),
                "max": float(np.max(baseline_precision))
            },
            "recall": {
                "mean": float(np.mean(baseline_recall)),
                "std": float(np.std(baseline_recall)),
                "min": float(np.min(baseline_recall)),
                "max": float(np.max(baseline_recall))
            },
            "fmeasure": {
                "mean": float(np.mean(baseline_fmeasure)),
                "std": float(np.std(baseline_fmeasure)),
                "min": float(np.min(baseline_fmeasure)),
                "max": float(np.max(baseline_fmeasure))
            }
        }
    else:
        aggregated["baseline"] = None
    
    if counterfactual_fmeasure:
        aggregated["counterfactual"] = {
            "precision": {
                "mean": float(np.mean(counterfactual_precision)),
                "std": float(np.std(counterfactual_precision)),
                "min": float(np.min(counterfactual_precision)),
                "max": float(np.max(counterfactual_precision))
            },
            "recall": {
                "mean": float(np.mean(counterfactual_recall)),
                "std": float(np.std(counterfactual_recall)),
                "min": float(np.min(counterfactual_recall)),
                "max": float(np.max(counterfactual_recall))
            },
            "fmeasure": {
                "mean": float(np.mean(counterfactual_fmeasure)),
                "std": float(np.std(counterfactual_fmeasure)),
                "min": float(np.min(counterfactual_fmeasure)),
                "max": float(np.max(counterfactual_fmeasure))
            }
        }
    else:
        aggregated["counterfactual"] = None
    
    return aggregated

def main():
    """
    Main function to process JSONL file and compute ROUGE-L scores.
    """
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description='Evaluate ROUGE-L scores for radiology reports.')
    parser.add_argument('input_file', type=str, help='Path to the input JSONL file')
    parser.add_argument('--output', type=str, default=None, help='Path for output JSONL file (default: rouge_results.jsonl)')
    parser.add_argument('--aggregated', type=str, default=None, help='Path for aggregated JSON file (default: rouge_results_aggregated.json)')
    args = parser.parse_args()
    
    jsonl_file_path = args.input_file
    output_file_path = args.output or 'rouge_results.jsonl'
    aggregated_file_path = args.aggregated or 'rouge_results_aggregated.json'
    
    print(f"Starting ROUGE-L evaluation of file: {jsonl_file_path}")
    print("=" * 60)
    
    all_results = []
    skipped_count = 0
    
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            for i, line in enumerate(infile):
                try:
                    # Load the JSON object from the line
                    data = json.loads(line)
                    
                    # Evaluate reports
                    results = evaluate_reports(data)
                    
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
        
        print("\n" + "=" * 60)
        print("ROUGE-L EVALUATION COMPLETE!")
        print("=" * 60)
        print(f"✓ Processed {len(all_results)} valid entries")
        print(f"✗ Skipped {skipped_count} invalid entries")
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
            print("\n" + "=" * 60)
            print("AGGREGATED ROUGE-L SCORES")
            print("=" * 60)
            print(f"Number of samples: {aggregated['n_samples']}")
            
            if aggregated['baseline']:
                print(f"\n--- Baseline Reports ---")
                print(f"Samples:   {aggregated['n_baseline_samples']}")
                print(f"Precision: {aggregated['baseline']['precision']['mean']:.4f} ± {aggregated['baseline']['precision']['std']:.4f}")
                print(f"Recall:    {aggregated['baseline']['recall']['mean']:.4f} ± {aggregated['baseline']['recall']['std']:.4f}")
                print(f"F-measure: {aggregated['baseline']['fmeasure']['mean']:.4f} ± {aggregated['baseline']['fmeasure']['std']:.4f}")
            else:
                print(f"\n--- Baseline Reports ---")
                print(f"No baseline scores available")
            
            if aggregated['counterfactual']:
                print(f"\n--- Counterfactual Reports ---")
                print(f"Samples:   {aggregated['n_counterfactual_samples']}")
                print(f"Precision: {aggregated['counterfactual']['precision']['mean']:.4f} ± {aggregated['counterfactual']['precision']['std']:.4f}")
                print(f"Recall:    {aggregated['counterfactual']['recall']['mean']:.4f} ± {aggregated['counterfactual']['recall']['std']:.4f}")
                print(f"F-measure: {aggregated['counterfactual']['fmeasure']['mean']:.4f} ± {aggregated['counterfactual']['fmeasure']['std']:.4f}")
            else:
                print(f"\n--- Counterfactual Reports ---")
                print(f"No counterfactual scores available")
            
            print("=" * 60)
        else:
            print("\n⚠ No valid results to aggregate!")
    
    except FileNotFoundError:
        print(f"Error: Input file not found at '{jsonl_file_path}'", file=sys.stderr)
        print("Please update the 'jsonl_file_path' variable in the script.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
