#!/usr/bin/env python3
"""
M2 LLJ Score Distance Calculator

Calculates the signed distance between LLJ scores for pure (original) and 
counterfactual questions. Also computes flip indicator for M3 correlation.

Distance = ((score_pure - 1) / 4) - ((score_cf - 1) / 4)  # Normalized to 0-1 range (scores are 1-5)

The script pairs entries by base ID (strips _pure/_counterfactual suffix),
computes the distance, and outputs augmented JSONL with `distance` and `flipped` fields.

Usage:
    python m2_llj_distance.py input.jsonl
    python m2_llj_distance.py input.jsonl -o output.jsonl --verbose
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def extract_base_id(entry_id: str) -> Tuple[str, str]:
    """
    Extract base ID and task type from entry_id.
    
    Args:
        entry_id: Entry ID like "123_pure" or "456_counterfactual"
        
    Returns:
        Tuple of (base_id, task_type)
    """
    if entry_id.endswith("_pure"):
        return entry_id[:-5], "pure"
    elif entry_id.endswith("_counterfactual"):
        return entry_id[:-15], "counterfactual"
    else:
        # Fallback: use task_type field if suffix not found
        return entry_id, "unknown"


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for flip comparison.
    
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
        match = re.match(r'^([A-E])[\s:]', answer)
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


def calculate_distance_and_flip(
    data: List[Dict],
    score_field: str = 'llm_judge_score',
    response_field: str = 'response',
    max_score: float = 5.0,
    verbose: bool = False
) -> Tuple[List[Dict], Dict]:
    """
    Calculate signed distance and flip indicator for each pair.
    
    Args:
        data: List of LLJ output entries
        score_field: Field name for LLJ score
        response_field: Field name for model response (for flip detection)
        max_score: Maximum score for normalization (default: 5, so range is 1-5)
        verbose: Print per-pair details
        
    Returns:
        Tuple of (augmented_entries, statistics)
    """
    # Group entries by base ID
    pairs = defaultdict(dict)
    
    for entry in data:
        entry_id = entry.get('entry_id', '')
        task_type = entry.get('task_type', '')
        
        # Determine task type from entry_id if not in field
        if not task_type:
            _, task_type = extract_base_id(entry_id)
        
        base_id, _ = extract_base_id(entry_id)
        pairs[base_id][task_type] = entry
    
    # Calculate distance and flip for each pair
    augmented = []
    stats = {
        'total_pairs': 0,
        'valid_pairs': 0,
        'skipped_null_score': 0,
        'skipped_missing_task': 0,
        'flipped_count': 0,
        'stable_count': 0,
        'distances': [],
        'absolute_distances': [],
        'pure_scores': [],
        'cf_scores': []
    }
    
    for base_id, pair in pairs.items():
        pure_entry = pair.get('pure')
        cf_entry = pair.get('counterfactual')
        
        # Skip if we don't have both
        if not pure_entry or not cf_entry:
            stats['skipped_missing_task'] += 1
            continue
        
        stats['total_pairs'] += 1
        
        # Get scores
        pure_score = pure_entry.get(score_field)
        cf_score = cf_entry.get(score_field)
        
        # Skip if either score is null
        if pure_score is None or cf_score is None:
            stats['skipped_null_score'] += 1
            if verbose:
                print(f"  SKIP (null score): {base_id}")
            continue
        
        stats['valid_pairs'] += 1
        
        # Calculate normalized distance: ((pure-1)/(max-1)) - ((cf-1)/(max-1))
        # This maps 1-5 scores to 0-1 range
        pure_norm_score = (pure_score - 1) / (max_score - 1)
        cf_norm_score = (cf_score - 1) / (max_score - 1)
        distance = pure_norm_score - cf_norm_score
        stats['distances'].append(distance)
        
        # Track absolute distance and raw scores for additional stats
        stats['absolute_distances'].append(abs(pure_score - cf_score))
        stats['pure_scores'].append(pure_score)
        stats['cf_scores'].append(cf_score)
        
        # Calculate flip indicator from responses
        pure_response = pure_entry.get(response_field, '')
        cf_response = cf_entry.get(response_field, '')
        
        pure_norm = normalize_answer(pure_response)
        cf_norm = normalize_answer(cf_response)
        
        flipped = 1 if pure_norm != cf_norm else 0
        
        if flipped:
            stats['flipped_count'] += 1
        else:
            stats['stable_count'] += 1
        
        if verbose and flipped:
            print(f"  FLIP: {base_id}: '{pure_response}' -> '{cf_response}' | distance={distance:.4f}")
        
        # Augment both entries with distance and flipped
        pure_augmented = pure_entry.copy()
        pure_augmented['distance'] = distance
        pure_augmented['flipped'] = flipped
        pure_augmented['pair_base_id'] = base_id
        
        cf_augmented = cf_entry.copy()
        cf_augmented['distance'] = distance
        cf_augmented['flipped'] = flipped
        cf_augmented['pair_base_id'] = base_id
        
        augmented.append(pure_augmented)
        augmented.append(cf_augmented)
    
    return augmented, stats


def main():
    parser = argparse.ArgumentParser(
        description='M2: Calculate LLJ score distance (pure - cf) and flip indicator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python m2_llj_distance.py llj_output.jsonl
    python m2_llj_distance.py llj_output.jsonl -o augmented.jsonl --verbose
        """
    )
    parser.add_argument('input_file', type=str, help='Path to LLJ output JSONL file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path for output JSONL file (default: input_m2.jsonl)')
    parser.add_argument('--score-field', type=str, default='llm_judge_score',
                        help='Field name for LLJ score (default: llm_judge_score)')
    parser.add_argument('--response-field', type=str, default='response',
                        help='Field name for model response (default: response)')
    parser.add_argument('--max-score', type=float, default=5.0,
                        help='Maximum score (default: 5.0, assumes min score is 1)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print details of flipped samples')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Default output path
    if args.output is None:
        input_stem = Path(args.input_file).stem
        args.output = f"{input_stem}_m2.jsonl"
    
    print(f"\n📊 M2 LLJ Score Distance Calculator")
    print("=" * 60)
    print(f"Input file:     {args.input_file}")
    print(f"Output file:    {args.output}")
    print(f"Score field:    {args.score_field}")
    print(f"Response field: {args.response_field}")
    print(f"Score range:    1 to {args.max_score}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} entries")
    
    # Calculate distance and flip
    print("\nCalculating distance and flip indicators...")
    if args.verbose:
        print("\nFlipped pairs:")
    
    augmented, stats = calculate_distance_and_flip(
        data,
        score_field=args.score_field,
        response_field=args.response_field,
        max_score=args.max_score,
        verbose=args.verbose
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 RESULTS")
    print("=" * 60)
    print(f"Total pairs found:       {stats['total_pairs']}")
    print(f"Valid pairs (both scores): {stats['valid_pairs']}")
    print(f"Skipped (null score):    {stats['skipped_null_score']}")
    print(f"Skipped (missing task):  {stats['skipped_missing_task']}")
    print("-" * 60)
    print(f"Flipped (response ≠):    {stats['flipped_count']}")
    print(f"Stable (response =):     {stats['stable_count']}")
    
    if stats['distances']:
        import statistics as st
        distances = stats['distances']
        print("-" * 60)
        print(f"Signed Distance (normalized) statistics:")
        print(f"  Mean:   {st.mean(distances):+.4f}")
        print(f"  Std:    {st.stdev(distances) if len(distances) > 1 else 0:.4f}")
        print(f"  Min:    {min(distances):+.4f}")
        print(f"  Max:    {max(distances):+.4f}")
        
        # Absolute distance statistics (raw scores)
        abs_distances = stats['absolute_distances']
        print("-" * 60)
        print(f"Absolute Distance (|pure - cf|) statistics:")
        print(f"  Mean:   {st.mean(abs_distances):.4f}")
        print(f"  Std:    {st.stdev(abs_distances) if len(abs_distances) > 1 else 0:.4f}")
        print(f"  Min:    {min(abs_distances):.4f}")
        print(f"  Max:    {max(abs_distances):.4f}")
        
        # Pearson correlation between pure and counterfactual scores
        pure_scores = stats['pure_scores']
        cf_scores = stats['cf_scores']
        if len(pure_scores) > 1:
            mean_pure = st.mean(pure_scores)
            mean_cf = st.mean(cf_scores)
            
            numerator = sum((p - mean_pure) * (c - mean_cf) for p, c in zip(pure_scores, cf_scores))
            denom_pure = sum((p - mean_pure) ** 2 for p in pure_scores) ** 0.5
            denom_cf = sum((c - mean_cf) ** 2 for c in cf_scores) ** 0.5
            
            if denom_pure > 0 and denom_cf > 0:
                correlation = numerator / (denom_pure * denom_cf)
                print("-" * 60)
                print(f"Pearson Correlation (pure vs cf scores): {correlation:.4f}")
            else:
                print("-" * 60)
                print(f"Pearson Correlation: undefined (no variance)")
    
    if stats['valid_pairs'] > 0:
        flip_rate = stats['flipped_count'] / stats['valid_pairs']
        print("-" * 60)
        print(f"Flip Rate: {flip_rate:.4f} ({flip_rate * 100:.2f}%)")
    print("=" * 60)
    
    # Save augmented output
    with open(args.output, 'w', encoding='utf-8') as f:
        for entry in augmented:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n✓ Augmented output saved to: {args.output}")
    print(f"  Total entries: {len(augmented)} (2 per valid pair)")


if __name__ == "__main__":
    main()
