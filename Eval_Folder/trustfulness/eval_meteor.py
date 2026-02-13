import json
import sys
import argparse
import re
import numpy as np
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize


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


def calculate_meteor(candidate, reference):
    """
    Calculates METEOR score for a candidate text against a reference text.
    
    Args:
        candidate: Generated text (string)
        reference: Ground truth text (string)
    
    Returns:
        METEOR score (float between 0 and 1)
    """
    if not candidate or not reference:
        return 0.0
    
    try:
        # Tokenize the texts
        candidate_tokens = word_tokenize(candidate.strip().lower())
        reference_tokens = word_tokenize(reference.strip().lower())
        
        # METEOR expects reference as a list of reference tokenizations
        references = [reference_tokens]
        
        # Calculate METEOR score
        score = meteor_score(references, candidate_tokens)
        return float(score)
    except Exception as e:
        print(f"Error calculating METEOR: {e}", file=sys.stderr)
        return 0.0

def evaluate_reports(data):
    """
    Evaluates METEOR scores for baseline and counterfactual reports.
    
    Args:
        data: Dictionary containing the reports
    
    Returns:
        Dictionary with METEOR scores or None if data is invalid
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
    
    # Calculate METEOR scores
    results = {}
    
    if baseline_response:
        results['baseline_meteor'] = calculate_meteor(baseline_response, ground_truth)
    else:
        results['baseline_meteor'] = None
    
    if counterfactual_response:
        results['counterfactual_meteor'] = calculate_meteor(counterfactual_response, ground_truth)
    else:
        results['counterfactual_meteor'] = None
    
    return results

def aggregate_results(all_results):
    """
    Aggregates METEOR scores across all samples.
    
    Args:
        all_results: List of result dictionaries
    
    Returns:
        Dictionary with aggregated statistics
    """
    if not all_results:
        print("No results to aggregate!")
        return None
    
    # Extract scores, filtering out None values
    baseline_scores = [r['baseline_meteor'] for r in all_results if r.get('baseline_meteor') is not None]
    counterfactual_scores = [r['counterfactual_meteor'] for r in all_results if r.get('counterfactual_meteor') is not None]
    
    aggregated = {
        "n_samples": len(all_results),
        "n_baseline_samples": len(baseline_scores),
        "n_counterfactual_samples": len(counterfactual_scores)
    }
    
    if baseline_scores:
        aggregated["baseline"] = {
            "mean": float(np.mean(baseline_scores)),
            "std": float(np.std(baseline_scores)),
            "min": float(np.min(baseline_scores)),
            "max": float(np.max(baseline_scores))
        }
    else:
        aggregated["baseline"] = None
    
    if counterfactual_scores:
        aggregated["counterfactual"] = {
            "mean": float(np.mean(counterfactual_scores)),
            "std": float(np.std(counterfactual_scores)),
            "min": float(np.min(counterfactual_scores)),
            "max": float(np.max(counterfactual_scores))
        }
    else:
        aggregated["counterfactual"] = None
    
    return aggregated

def main():
    """
    Main function to process JSONL file and compute METEOR scores.
    """
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description='Evaluate METEOR scores for radiology reports.')
    parser.add_argument('input_file', type=str, help='Path to the input JSONL file')
    parser.add_argument('--output', type=str, default=None, help='Path for output JSONL file (default: meteor_results.jsonl)')
    parser.add_argument('--aggregated', type=str, default=None, help='Path for aggregated JSON file (default: meteor_results_aggregated.json)')
    args = parser.parse_args()
    
    jsonl_file_path = args.input_file
    output_file_path = args.output or 'meteor_results.jsonl'
    aggregated_file_path = args.aggregated or 'meteor_results_aggregated.json'
    
    print(f"Starting METEOR evaluation of file: {jsonl_file_path}")
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
        print("METEOR EVALUATION COMPLETE!")
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
            print("AGGREGATED METEOR SCORES")
            print("=" * 60)
            print(f"Number of samples: {aggregated['n_samples']}")
            
            if aggregated['baseline']:
                print(f"\n--- Baseline Reports ---")
                print(f"Samples: {aggregated['n_baseline_samples']}")
                print(f"Mean:    {aggregated['baseline']['mean']:.4f} ± {aggregated['baseline']['std']:.4f}")
                print(f"Min:     {aggregated['baseline']['min']:.4f}")
                print(f"Max:     {aggregated['baseline']['max']:.4f}")
            else:
                print(f"\n--- Baseline Reports ---")
                print(f"No baseline scores available")
            
            if aggregated['counterfactual']:
                print(f"\n--- Counterfactual Reports ---")
                print(f"Samples: {aggregated['n_counterfactual_samples']}")
                print(f"Mean:    {aggregated['counterfactual']['mean']:.4f} ± {aggregated['counterfactual']['std']:.4f}")
                print(f"Min:     {aggregated['counterfactual']['min']:.4f}")
                print(f"Max:     {aggregated['counterfactual']['max']:.4f}")
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
