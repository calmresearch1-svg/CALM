#!/usr/bin/env python3
"""
M1 BERTScore Similarity Calculator for Report Generation

Calculates BERTScore similarity between original (pure) and counterfactual responses.
This measures how much the model's response changed when the input was modified.

Higher similarity = responses are similar (stable)
Lower similarity = responses are different (like a "flip" for open-ended text)

Works with LLJ output files that have entry_id with _pure/_counterfactual suffix.

Usage:
    python m1_bertscore.py llj_output.jsonl
    python m1_bertscore.py llj_output.jsonl --augment -o augmented.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

try:
    from bert_score import score
except ImportError:
    print("Error: bert_score package not installed. Install with: pip install bert-score", file=sys.stderr)
    sys.exit(1)


def calculate_bertscore(
    candidates: List[str],
    references: List[str],
    model_type: str = 'roberta-large',
    lang: str = 'en'
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculates BERTScore for candidate texts against reference texts.
    
    Returns:
        Tuple of (precision, recall, f1) lists
    """
    if not candidates or not references:
        return [], [], []
    
    try:
        P, R, F1 = score(
            candidates, 
            references, 
            model_type=model_type,
            lang=lang,
            verbose=True,
            rescale_with_baseline=False
        )
        
        precision = [float(p) for p in P.tolist()]
        recall = [float(r) for r in R.tolist()]
        f1 = [float(f) for f in F1.tolist()]
        
        return precision, recall, f1
    except Exception as e:
        print(f"Error calculating BERTScore: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return [], [], []


def extract_base_id(entry_id: str) -> Tuple[str, str]:
    """Extract base ID and task type from entry_id."""
    if entry_id.endswith("_pure"):
        return entry_id[:-5], "pure"
    elif entry_id.endswith("_counterfactual"):
        return entry_id[:-15], "counterfactual"
    else:
        return entry_id, "unknown"


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
    return data


def pair_entries(
    data: List[Dict],
    response_field: str = 'response'
) -> Tuple[List[Dict], Dict]:
    """
    Pair pure and counterfactual entries by base ID.
    
    Returns:
        Tuple of (pairs_list, statistics)
    """
    # Group by base ID
    groups = defaultdict(dict)
    
    for entry in data:
        entry_id = entry.get('entry_id', '')
        task_type = entry.get('task_type', '')
        
        if not task_type:
            _, task_type = extract_base_id(entry_id)
        
        base_id, _ = extract_base_id(entry_id)
        groups[base_id][task_type] = entry
    
    # Build pairs
    pairs = []
    stats = {
        'total_entries': len(data),
        'unique_base_ids': len(groups),
        'valid_pairs': 0,
        'missing_pure': 0,
        'missing_cf': 0,
        'missing_response': 0,
    }
    
    for base_id, group in groups.items():
        pure_entry = group.get('pure')
        cf_entry = group.get('counterfactual')
        
        if not pure_entry:
            stats['missing_pure'] += 1
            continue
        if not cf_entry:
            stats['missing_cf'] += 1
            continue
        
        pure_response = pure_entry.get(response_field, '')
        cf_response = cf_entry.get(response_field, '')
        
        if not pure_response or not cf_response:
            stats['missing_response'] += 1
            continue
        
        pairs.append({
            'base_id': base_id,
            'pure_entry': pure_entry,
            'cf_entry': cf_entry,
            'pure_response': pure_response,
            'cf_response': cf_response,
        })
        stats['valid_pairs'] += 1
    
    return pairs, stats


def main():
    parser = argparse.ArgumentParser(
        description='M1: Calculate BERTScore similarity between pure and counterfactual responses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python m1_bertscore.py llj_output.jsonl
    python m1_bertscore.py llj_output.jsonl --augment -o augmented.jsonl
    
Output:
    - bertscore_similarity: F1 score between pure and cf responses (0-1)
    - Higher = more similar (stable), Lower = more different (changed)
        """
    )
    parser.add_argument('input_file', type=str, help='Path to LLJ output JSONL file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path for output file')
    parser.add_argument('--augment', action='store_true',
                        help='Output augmented JSONL with bertscore_similarity for M3')
    parser.add_argument('--response-field', type=str, default='response',
                        help='Field name for model response (default: response)')
    parser.add_argument('--model', '-m', type=str, default='roberta-large',
                        help='Model for BERTScore (default: roberta-large)')
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    input_stem = Path(args.input_file).stem
    
    print(f"\n📊 M1 BERTScore Similarity Calculator")
    print("=" * 60)
    print(f"Input file:     {args.input_file}")
    print(f"Response field: {args.response_field}")
    print(f"Model:          {args.model}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} entries")
    
    # Pair entries
    print("\nPairing pure and counterfactual entries...")
    pairs, stats = pair_entries(data, response_field=args.response_field)
    
    print(f"Valid pairs: {stats['valid_pairs']}")
    if stats['missing_pure'] > 0:
        print(f"  Missing pure: {stats['missing_pure']}")
    if stats['missing_cf'] > 0:
        print(f"  Missing cf: {stats['missing_cf']}")
    if stats['missing_response'] > 0:
        print(f"  Missing response: {stats['missing_response']}")
    
    if not pairs:
        print("Error: No valid pairs found!", file=sys.stderr)
        sys.exit(1)
    
    # Calculate BERTScore between pure and cf responses
    print(f"\nCalculating BERTScore similarity for {len(pairs)} pairs...")
    pure_responses = [p['pure_response'] for p in pairs]
    cf_responses = [p['cf_response'] for p in pairs]
    
    P, R, F1 = calculate_bertscore(pure_responses, cf_responses, args.model)
    
    if not F1:
        print("Error: BERTScore calculation failed!", file=sys.stderr)
        sys.exit(1)
    
    # Add scores to pairs
    for i, pair in enumerate(pairs):
        pair['bertscore_precision'] = P[i]
        pair['bertscore_recall'] = R[i]
        pair['bertscore_similarity'] = F1[i]  # F1 is our similarity measure
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 RESULTS")
    print("=" * 60)
    print(f"Valid pairs: {len(pairs)}")
    print("-" * 60)
    print(f"BERTScore Similarity (pure vs cf):")
    print(f"  Mean:   {np.mean(F1):.4f}")
    print(f"  Std:    {np.std(F1):.4f}")
    print(f"  Min:    {min(F1):.4f}")
    print(f"  Max:    {max(F1):.4f}")
    print("=" * 60)
    
    # Output
    if args.augment:
        output_path = args.output or f'{input_stem}_m1.jsonl'
        
        # Build lookup by base_id
        similarity_lookup = {p['base_id']: p['bertscore_similarity'] for p in pairs}
        
        # Write augmented entries (both pure and cf get the same similarity score)
        augmented_count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                entry_id = entry.get('entry_id', '')
                base_id, _ = extract_base_id(entry_id)
                
                if base_id in similarity_lookup:
                    augmented = entry.copy()
                    augmented['bertscore_similarity'] = similarity_lookup[base_id]
                    augmented['pair_base_id'] = base_id
                    f.write(json.dumps(augmented) + '\n')
                    augmented_count += 1
        
        print(f"\n✓ Augmented JSONL saved to: {output_path}")
        print(f"  Entries: {augmented_count}")
        print(f"  Added 'bertscore_similarity' field (0-1, higher = more similar)")
    
    else:
        # Default: save per-pair results
        output_path = args.output or f'{input_stem}_bertscore.jsonl'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                result = {
                    'base_id': pair['base_id'],
                    'bertscore_precision': pair['bertscore_precision'],
                    'bertscore_recall': pair['bertscore_recall'],
                    'bertscore_similarity': pair['bertscore_similarity'],
                }
                f.write(json.dumps(result) + '\n')
        
        # Also save aggregated summary
        agg_path = f'{input_stem}_bertscore_aggregated.json'
        aggregated = {
            'input_file': str(args.input_file),
            'model': args.model,
            'n_pairs': len(pairs),
            'similarity': {
                'mean': float(np.mean(F1)),
                'std': float(np.std(F1)),
                'min': float(min(F1)),
                'max': float(max(F1)),
            }
        }
        with open(agg_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"\n✓ Per-pair results saved to: {output_path}")
        print(f"✓ Aggregated summary saved to: {agg_path}")


if __name__ == "__main__":
    main()
