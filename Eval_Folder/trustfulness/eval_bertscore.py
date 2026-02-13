import json
import sys
import argparse
import re
import numpy as np
from bert_score import score


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


def calculate_bertscore(candidates, references, model_type='roberta-large', lang='en'):
    """
    Calculates BERTScore for candidate texts against reference texts.
    
    Args:
        candidates: List of generated texts
        references: List of ground truth texts
        model_type: Pre-trained model to use for BERTScore
        lang: Language code (default: 'en' for English)
    
    Returns:
        Tuple of (precision, recall, f1) lists
    """
    if not candidates or not references:
        return [], [], []
    
    try:
        # Calculate BERTScore
        # Returns precision, recall, and F1 tensors
        # Note: rescale_with_baseline=False keeps scores in 0-1 range
        P, R, F1 = score(
            candidates, 
            references, 
            model_type=model_type,
            lang=lang,
            verbose=True,
            rescale_with_baseline=False
        )
        
        # Convert tensors to lists of floats
        precision = [float(p) for p in P.tolist()]
        recall = [float(r) for r in R.tolist()]
        f1 = [float(f) for f in F1.tolist()]
        
        return precision, recall, f1
    except Exception as e:
        print(f"Error calculating BERTScore: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return [], [], []

def evaluate_reports(data):
    """
    Extracts baseline and counterfactual reports for BERTScore evaluation.
    
    Args:
        data: Dictionary containing the reports
    
    Returns:
        Dictionary with report texts or None if data is invalid
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
    
    # Clean all texts before returning
    ground_truth = clean_report(ground_truth)
    baseline_response = clean_report(baseline_response) if baseline_response else None
    counterfactual_response = clean_report(counterfactual_response) if counterfactual_response else None
    
    return {
        'ground_truth': ground_truth,
        'baseline': baseline_response,
        'counterfactual': counterfactual_response
    }

def aggregate_results(all_results):
    """
    Aggregates BERTScore results across all samples.
    
    Args:
        all_results: List of result dictionaries
    
    Returns:
        Dictionary with aggregated statistics
    """
    if not all_results:
        print("No results to aggregate!")
        return None
    
    # Extract scores, filtering out None values
    baseline_precision = [r['baseline_bertscore']['precision'] for r in all_results 
                         if r.get('baseline_bertscore') is not None]
    baseline_recall = [r['baseline_bertscore']['recall'] for r in all_results 
                      if r.get('baseline_bertscore') is not None]
    baseline_f1 = [r['baseline_bertscore']['f1'] for r in all_results 
                  if r.get('baseline_bertscore') is not None]
    
    counterfactual_precision = [r['counterfactual_bertscore']['precision'] for r in all_results 
                               if r.get('counterfactual_bertscore') is not None]
    counterfactual_recall = [r['counterfactual_bertscore']['recall'] for r in all_results 
                            if r.get('counterfactual_bertscore') is not None]
    counterfactual_f1 = [r['counterfactual_bertscore']['f1'] for r in all_results 
                        if r.get('counterfactual_bertscore') is not None]
    
    aggregated = {
        "n_samples": len(all_results),
        "n_baseline_samples": len(baseline_f1),
        "n_counterfactual_samples": len(counterfactual_f1)
    }
    
    if baseline_f1:
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
            "f1": {
                "mean": float(np.mean(baseline_f1)),
                "std": float(np.std(baseline_f1)),
                "min": float(np.min(baseline_f1)),
                "max": float(np.max(baseline_f1))
            }
        }
    else:
        aggregated["baseline"] = None
    
    if counterfactual_f1:
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
            "f1": {
                "mean": float(np.mean(counterfactual_f1)),
                "std": float(np.std(counterfactual_f1)),
                "min": float(np.min(counterfactual_f1)),
                "max": float(np.max(counterfactual_f1))
            }
        }
    else:
        aggregated["counterfactual"] = None
    
    return aggregated

def main():
    """
    Main function to process JSONL file and compute BERTScores.
    """
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description='Evaluate BERTScores for radiology reports.')
    parser.add_argument('input_file', type=str, help='Path to the input JSONL file')
    parser.add_argument('--output', type=str, default=None, help='Path for output JSONL file (default: bertscore_results.jsonl)')
    parser.add_argument('--aggregated', type=str, default=None, help='Path for aggregated JSON file (default: bertscore_results_aggregated.json)')
    parser.add_argument('--model', type=str, default='roberta-large',
                        help='Model to use for BERTScore. Recommended: roberta-large (default), distilbert-base-uncased, bert-base-uncased')
    args = parser.parse_args()
    
    jsonl_file_path = args.input_file
    output_file_path = args.output or 'bertscore_results.jsonl'
    aggregated_file_path = args.aggregated or 'bertscore_results_aggregated.json'
    model_type = args.model
    
    print(f"Starting BERTScore evaluation of file: {jsonl_file_path}")
    print(f"Using model: {model_type}")
    print("=" * 60)
    
    all_data = []
    skipped_count = 0
    
    try:
        # First pass: collect all data
        with open(jsonl_file_path, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                try:
                    # Load the JSON object from the line
                    data = json.loads(line)
                    
                    # Extract reports
                    reports = evaluate_reports(data)
                    
                    if reports:
                        all_data.append(reports)
                    else:
                        skipped_count += 1
                        
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {i+1}: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing line {i+1}: {e}", file=sys.stderr)
        
        if not all_data:
            print("\n⚠ No valid data found!")
            return
        
        print(f"\nCollected {len(all_data)} valid entries")
        print(f"Skipped {skipped_count} invalid entries")
        print("\nCalculating BERTScores (this may take a while)...")
        
        # Separate baseline and counterfactual for batch processing
        baseline_candidates = []
        baseline_references = []
        baseline_indices = []
        
        counterfactual_candidates = []
        counterfactual_references = []
        counterfactual_indices = []
        
        for idx, item in enumerate(all_data):
            if item['baseline']:
                baseline_candidates.append(item['baseline'])
                baseline_references.append(item['ground_truth'])
                baseline_indices.append(idx)
            
            if item['counterfactual']:
                counterfactual_candidates.append(item['counterfactual'])
                counterfactual_references.append(item['ground_truth'])
                counterfactual_indices.append(idx)
        
        # Calculate BERTScores in batch
        all_results = [None] * len(all_data)
        
        if baseline_candidates:
            print(f"\nProcessing {len(baseline_candidates)} baseline reports...")
            baseline_P, baseline_R, baseline_F1 = calculate_bertscore(
                baseline_candidates, baseline_references, model_type
            )
            
            for i, idx in enumerate(baseline_indices):
                if all_results[idx] is None:
                    all_results[idx] = {}
                all_results[idx]['baseline_bertscore'] = {
                    'precision': baseline_P[i],
                    'recall': baseline_R[i],
                    'f1': baseline_F1[i]
                }
        
        if counterfactual_candidates:
            print(f"Processing {len(counterfactual_candidates)} counterfactual reports...")
            cf_P, cf_R, cf_F1 = calculate_bertscore(
                counterfactual_candidates, counterfactual_references, model_type
            )
            
            for i, idx in enumerate(counterfactual_indices):
                if all_results[idx] is None:
                    all_results[idx] = {}
                all_results[idx]['counterfactual_bertscore'] = {
                    'precision': cf_P[i],
                    'recall': cf_R[i],
                    'f1': cf_F1[i]
                }
        
        # Filter out None results
        all_results = [r for r in all_results if r is not None]
        
        # Write individual results
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for result in all_results:
                json.dump(result, outfile)
                outfile.write('\n')
        
        print("\n" + "=" * 60)
        print("BERTSCORE EVALUATION COMPLETE!")
        print("=" * 60)
        print(f"✓ Processed {len(all_results)} valid entries")
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
            print("AGGREGATED BERTSCORES")
            print("=" * 60)
            print(f"Model: {model_type}")
            print(f"Number of samples: {aggregated['n_samples']}")
            
            if aggregated['baseline']:
                print(f"\n--- Baseline Reports ---")
                print(f"Samples:   {aggregated['n_baseline_samples']}")
                print(f"Precision: {aggregated['baseline']['precision']['mean']:.4f} ± {aggregated['baseline']['precision']['std']:.4f}")
                print(f"Recall:    {aggregated['baseline']['recall']['mean']:.4f} ± {aggregated['baseline']['recall']['std']:.4f}")
                print(f"F1:        {aggregated['baseline']['f1']['mean']:.4f} ± {aggregated['baseline']['f1']['std']:.4f}")
            else:
                print(f"\n--- Baseline Reports ---")
                print(f"No baseline scores available")
            
            if aggregated['counterfactual']:
                print(f"\n--- Counterfactual Reports ---")
                print(f"Samples:   {aggregated['n_counterfactual_samples']}")
                print(f"Precision: {aggregated['counterfactual']['precision']['mean']:.4f} ± {aggregated['counterfactual']['precision']['std']:.4f}")
                print(f"Recall:    {aggregated['counterfactual']['recall']['mean']:.4f} ± {aggregated['counterfactual']['recall']['std']:.4f}")
                print(f"F1:        {aggregated['counterfactual']['f1']['mean']:.4f} ± {aggregated['counterfactual']['f1']['std']:.4f}")
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
