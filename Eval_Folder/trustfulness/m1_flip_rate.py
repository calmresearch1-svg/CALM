#!/usr/bin/env python3
"""
M1 Flip Rate Calculator for Phase 1 Evaluation

Calculates the flip rate (stability rate) for binary (TF) and multiple choice (MC) questions.
Flip Rate = Pr(OO = CO) - the rate at which Original Outcome equals Counterfactual Outcome.

Usage:
    python m1_flip_rate.py input.jsonl
    python m1_flip_rate.py input.jsonl --output results.json
    python m1_flip_rate.py input.jsonl --verbose
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.
    
    - Strips whitespace
    - Converts to uppercase
    - Extracts first letter if answer contains option format (e.g., "A: something" -> "A")
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer string
    """
    if not answer:
        return ""
    
    answer = str(answer).strip().upper()
    
    # If answer is multiple characters and starts with a letter followed by colon or space,
    # extract just the letter (e.g., "A: some text" -> "A", "B something" -> "B")
    if len(answer) > 1:
        match = re.match(r'^([A-E])[:\s]', answer)
        if match:
            return match.group(1)
    
    # If it's just a single letter A-E, return it
    if len(answer) == 1 and answer in 'ABCDE':
        return answer
    
    # For yes/no (TF) questions, normalize common variations
    answer_lower = answer.lower()
    if answer_lower in ['yes', 'y', 'true', 't', '1']:
        return 'YES'
    if answer_lower in ['no', 'n', 'false', 'f', '0']:
        return 'NO'
    
    return answer


def calculate_flip_rate(
    data: List[Dict],
    pure_field: str = 'pure_prediction',
    cf_field: str = 'cf_prediction',
    verbose: bool = False
) -> Dict:
    """
    Calculate flip rate statistics.
    
    Args:
        data: List of dictionaries containing predictions
        pure_field: Field name for original/pure prediction
        cf_field: Field name for counterfactual prediction
        verbose: Whether to print per-sample details
        
    Returns:
        Dictionary with flip rate statistics
    """
    total = 0
    matches = 0
    mismatches = 0
    skipped = 0
    per_sample_results = []
    
    for i, entry in enumerate(data):
        pure_pred = entry.get(pure_field)
        cf_pred = entry.get(cf_field)
        
        # Skip entries without both predictions
        if pure_pred is None or cf_pred is None:
            skipped += 1
            continue
        
        # Normalize answers
        pure_norm = normalize_answer(pure_pred)
        cf_norm = normalize_answer(cf_pred)
        
        total += 1
        is_match = pure_norm == cf_norm
        
        if is_match:
            matches += 1
        else:
            mismatches += 1
        
        sample_result = {
            'index': i,
            'pure_original': pure_pred,
            'cf_original': cf_pred,
            'pure_normalized': pure_norm,
            'cf_normalized': cf_norm,
            'is_stable': is_match
        }
        
        # Add question_id if available
        if 'question_id' in entry:
            sample_result['question_id'] = entry['question_id']
        
        per_sample_results.append(sample_result)
        
        if verbose and not is_match:
            q_id = entry.get('question_id', f'sample_{i}')
            print(f"  FLIP: {q_id}: '{pure_pred}' -> '{cf_pred}' (normalized: '{pure_norm}' vs '{cf_norm}')")
    
    # Calculate rates
    stability_rate = matches / total if total > 0 else 0.0
    flip_rate = mismatches / total if total > 0 else 0.0
    
    return {
        'total_samples': total,
        'skipped_samples': skipped,
        'stable_count': matches,
        'flipped_count': mismatches,
        'stability_rate': stability_rate,
        'flip_rate': flip_rate,
        'stability_rate_percent': stability_rate * 100,
        'flip_rate_percent': flip_rate * 100,
        'per_sample_results': per_sample_results
    }


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


def main():
    parser = argparse.ArgumentParser(
        description='Calculate flip rate for TF/MC questions (M1 metric)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python m1_flip_rate.py data/phase1_results/qwen_predictions_ham10000_filtered.jsonl
    python m1_flip_rate.py input.jsonl --output flip_rate_results.json --verbose
    python m1_flip_rate.py input.jsonl --augment -o augmented.jsonl  # Add flipped field
        """
    )
    parser.add_argument('input_file', type=str, help='Path to input JSONL file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path for output file (JSON summary or augmented JSONL)')
    parser.add_argument('--augment', '-a', action='store_true',
                        help='Output augmented JSONL with flipped field (0=stable, 1=flipped)')
    parser.add_argument('--pure-field', type=str, default='pure_prediction',
                        help='Field name for original/pure prediction (default: pure_prediction)')
    parser.add_argument('--cf-field', type=str, default='cf_prediction',
                        help='Field name for counterfactual prediction (default: cf_prediction)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print details of flipped samples')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n📊 M1 Flip Rate Calculator")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Pure field: {args.pure_field}")
    print(f"CF field:   {args.cf_field}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} entries from file")
    
    # Calculate flip rate
    print("\nCalculating flip rate...")
    if args.verbose:
        print("\nFlipped samples:")
    
    results = calculate_flip_rate(
        data,
        pure_field=args.pure_field,
        cf_field=args.cf_field,
        verbose=args.verbose
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 RESULTS")
    print("=" * 60)
    print(f"Total samples analyzed:  {results['total_samples']}")
    print(f"Samples skipped:         {results['skipped_samples']}")
    print(f"Stable (OO = CO):        {results['stable_count']}")
    print(f"Flipped (OO ≠ CO):       {results['flipped_count']}")
    print("-" * 60)
    print(f"Stability Rate Pr(OO=CO): {results['stability_rate']:.4f} ({results['stability_rate_percent']:.2f}%)")
    print(f"Flip Rate Pr(OO≠CO):      {results['flip_rate']:.4f} ({results['flip_rate_percent']:.2f}%)")
    print("=" * 60)
    
    # Save results based on mode
    if args.augment:
        # Augment mode: output JSONL with flipped field added to each entry
        output_path = args.output or f"{Path(args.input_file).stem}_m1.jsonl"
        
        # Build lookup from per_sample_results
        flip_lookup = {}
        for sample in results['per_sample_results']:
            idx = sample['index']
            flip_lookup[idx] = 1 if not sample['is_stable'] else 0
        
        # Write augmented entries
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(data):
                augmented = entry.copy()
                if i in flip_lookup:
                    augmented['flipped'] = flip_lookup[i]
                f.write(json.dumps(augmented) + '\n')
        
        print(f"\n✓ Augmented JSONL saved to: {output_path}")
        print(f"  Added 'flipped' field (0=stable, 1=flipped) to {len(flip_lookup)} entries")
    
    elif args.output:
        # Default mode with output: save JSON summary
        output_data = {
            'input_file': str(args.input_file),
            'pure_field': args.pure_field,
            'cf_field': args.cf_field,
            'summary': {
                'total_samples': results['total_samples'],
                'skipped_samples': results['skipped_samples'],
                'stable_count': results['stable_count'],
                'flipped_count': results['flipped_count'],
                'stability_rate': results['stability_rate'],
                'flip_rate': results['flip_rate']
            },
            'per_sample_results': results['per_sample_results']
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
