import json
import sys
import argparse
import re
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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


def calculate_bleu(candidate, reference):
    """
    Calculates BLEU scores (1-4 grams) for a candidate text against a reference text.
    
    Args:
        candidate: Generated text (string)
        reference: Ground truth text (string)
    
    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    if not candidate or not reference:
        return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
    
    # Tokenize the texts (convert to lowercase for better matching)
    candidate_tokens = candidate.strip().lower().split()
    reference_tokens = reference.strip().lower().split()
    
    # BLEU expects reference as a list of reference tokenizations
    references = [reference_tokens]
    
    # Use smoothing method 4 (best for sentence-level BLEU)
    smoothing = SmoothingFunction().method4
    
    try:
        # Calculate individual n-gram BLEU scores
        bleu_1 = sentence_bleu(references, candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu_2 = sentence_bleu(references, candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu_3 = sentence_bleu(references, candidate_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoothing)
        bleu_4 = sentence_bleu(references, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        return {
            'bleu_1': float(bleu_1),
            'bleu_2': float(bleu_2),
            'bleu_3': float(bleu_3),
            'bleu_4': float(bleu_4)
        }
    except Exception as e:
        print(f"Error calculating BLEU: {e}", file=sys.stderr)
        return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}

def evaluate_reports(data):
    """
    Evaluates BLEU scores for baseline and counterfactual reports.
    
    Args:
        data: Dictionary containing the reports
    
    Returns:
        Dictionary with BLEU scores or None if data is invalid
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
    
    # Calculate BLEU scores
    results = {}
    
    if baseline_response:
        results['baseline_bleu'] = calculate_bleu(baseline_response, ground_truth)
    else:
        results['baseline_bleu'] = None
    
    if counterfactual_response:
        results['counterfactual_bleu'] = calculate_bleu(counterfactual_response, ground_truth)
    else:
        results['counterfactual_bleu'] = None
    
    return results

def aggregate_results(all_results):
    """
    Aggregates BLEU scores across all samples.
    
    Args:
        all_results: List of result dictionaries
    
    Returns:
        Dictionary with aggregated statistics
    """
    if not all_results:
        print("No results to aggregate!")
        return None
    
    # Extract scores, filtering out None values
    baseline_scores = [r['baseline_bleu'] for r in all_results if r.get('baseline_bleu') is not None]
    counterfactual_scores = [r['counterfactual_bleu'] for r in all_results if r.get('counterfactual_bleu') is not None]
    
    aggregated = {
        "n_samples": len(all_results),
        "n_baseline_samples": len(baseline_scores),
        "n_counterfactual_samples": len(counterfactual_scores)
    }
    
    # Helper function to aggregate a specific BLEU-n score
    def aggregate_bleu_n(scores_list, key):
        values = [s[key] for s in scores_list]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }
    
    if baseline_scores:
        aggregated["baseline"] = {
            "bleu_1": aggregate_bleu_n(baseline_scores, 'bleu_1'),
            "bleu_2": aggregate_bleu_n(baseline_scores, 'bleu_2'),
            "bleu_3": aggregate_bleu_n(baseline_scores, 'bleu_3'),
            "bleu_4": aggregate_bleu_n(baseline_scores, 'bleu_4')
        }
    else:
        aggregated["baseline"] = None
    
    if counterfactual_scores:
        aggregated["counterfactual"] = {
            "bleu_1": aggregate_bleu_n(counterfactual_scores, 'bleu_1'),
            "bleu_2": aggregate_bleu_n(counterfactual_scores, 'bleu_2'),
            "bleu_3": aggregate_bleu_n(counterfactual_scores, 'bleu_3'),
            "bleu_4": aggregate_bleu_n(counterfactual_scores, 'bleu_4')
        }
    else:
        aggregated["counterfactual"] = None
    
    return aggregated

def main():
    """
    Main function to process JSONL file and compute BLEU scores.
    """
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description='Evaluate BLEU scores for radiology reports.')
    parser.add_argument('input_file', type=str, help='Path to the input JSONL file')
    parser.add_argument('--output', type=str, default=None, help='Path for output JSONL file (default: bleu_results.jsonl)')
    parser.add_argument('--aggregated', type=str, default=None, help='Path for aggregated JSON file (default: bleu_results_aggregated.json)')
    args = parser.parse_args()
    
    jsonl_file_path = args.input_file
    output_file_path = args.output or 'bleu_results.jsonl'
    aggregated_file_path = args.aggregated or 'bleu_results_aggregated.json'
    
    print(f"Starting BLEU evaluation of file: {jsonl_file_path}")
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
        print("BLEU EVALUATION COMPLETE!")
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
            print("AGGREGATED BLEU SCORES")
            print("=" * 60)
            print(f"Number of samples: {aggregated['n_samples']}")
            
            if aggregated['baseline']:
                print(f"\n--- Baseline Reports ---")
                print(f"Samples: {aggregated['n_baseline_samples']}")
                print(f"BLEU-1:  {aggregated['baseline']['bleu_1']['mean']:.4f} ± {aggregated['baseline']['bleu_1']['std']:.4f}")
                print(f"BLEU-2:  {aggregated['baseline']['bleu_2']['mean']:.4f} ± {aggregated['baseline']['bleu_2']['std']:.4f}")
                print(f"BLEU-3:  {aggregated['baseline']['bleu_3']['mean']:.4f} ± {aggregated['baseline']['bleu_3']['std']:.4f}")
                print(f"BLEU-4:  {aggregated['baseline']['bleu_4']['mean']:.4f} ± {aggregated['baseline']['bleu_4']['std']:.4f}")
            else:
                print(f"\n--- Baseline Reports ---")
                print(f"No baseline scores available")
            
            if aggregated['counterfactual']:
                print(f"\n--- Counterfactual Reports ---")
                print(f"Samples: {aggregated['n_counterfactual_samples']}")
                print(f"BLEU-1:  {aggregated['counterfactual']['bleu_1']['mean']:.4f} ± {aggregated['counterfactual']['bleu_1']['std']:.4f}")
                print(f"BLEU-2:  {aggregated['counterfactual']['bleu_2']['mean']:.4f} ± {aggregated['counterfactual']['bleu_2']['std']:.4f}")
                print(f"BLEU-3:  {aggregated['counterfactual']['bleu_3']['mean']:.4f} ± {aggregated['counterfactual']['bleu_3']['std']:.4f}")
                print(f"BLEU-4:  {aggregated['counterfactual']['bleu_4']['mean']:.4f} ± {aggregated['counterfactual']['bleu_4']['std']:.4f}")
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
