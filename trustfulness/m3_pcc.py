#!/usr/bin/env python3
"""
M3 Pearson Correlation Coefficient (PCC) Calculator

Calculates the Pearson correlation between flip indicator (binary 0/1) and 
LLJ score distance. This measures whether samples that flip their answers
also show larger discrepancies in LLJ judge scores.

Expected input: JSONL with 'flipped' (0/1) and 'distance' (float) fields,
typically output from m2_llj_distance.py.

Usage:
    python m3_pcc.py input.jsonl
    python m3_pcc.py input.jsonl --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from scipy import stats
    import numpy as np
except ImportError:
    print("Error: scipy and numpy required. Install with: pip install scipy numpy", file=sys.stderr)
    sys.exit(1)


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


def extract_pairs(
    data: List[Dict],
    flipped_field: str = 'flipped',
    distance_field: str = 'distance'
) -> Tuple[List[float], List[float], int]:
    """
    Extract M1 and M2 (distance) values, deduplicating by pair.
    
    M1 can be binary (flipped: 0/1) or continuous (bertscore_distance).
    
    Since M2 outputs 2 entries per pair (pure and counterfactual) with same 
    values, we deduplicate by pair_base_id if available.
    
    Returns:
        Tuple of (m1_values, m2_values, skipped_count)
    """
    seen_pairs = set()
    m1_values = []
    m2_values = []
    skipped = 0
    
    for entry in data:
        # Get pair identifier (deduplicate)
        pair_id = entry.get('pair_base_id')
        if pair_id:
            if pair_id in seen_pairs:
                continue
            seen_pairs.add(pair_id)
        
        flipped = entry.get(flipped_field)
        distance = entry.get(distance_field)
        
        # Skip if missing values
        if flipped is None or distance is None:
            skipped += 1
            continue
        
        m1_values.append(float(flipped))  # Works for both binary and continuous
        m2_values.append(float(distance))
    
    return m1_values, m2_values, skipped


def calculate_correlation(
    m1_values: List[float],
    m2_values: List[float],
    method: str = 'auto'
) -> Dict:
    """
    Calculate correlation between M1 and M2 values.
    
    Args:
        m1_values: List of M1 values (can be binary or continuous)
        m2_values: List of M2 values (continuous)
        method: 'pearson', 'pointbiserial', or 'auto' (auto-detects binary M1)
    
    Returns:
        Dictionary with correlation results
    """
    if len(m1_values) < 2:
        return {
            'correlation': None,
            'p_value': None,
            'n_samples': len(m1_values),
            'method': method,
            'error': 'Need at least 2 samples for correlation'
        }
    
    # Check for zero variance
    if np.std(m1_values) == 0:
        return {
            'correlation': None,
            'p_value': None,
            'n_samples': len(m1_values),
            'method': method,
            'error': 'Zero variance in M1 values (all same)'
        }
    
    if np.std(m2_values) == 0:
        return {
            'correlation': None,
            'p_value': None,
            'n_samples': len(m1_values),
            'method': method,
            'error': 'Zero variance in M2 values (all same)'
        }
    
    # Determine if M1 is binary
    unique_m1 = set(m1_values)
    is_binary = unique_m1 <= {0.0, 1.0}
    
    # Auto-select method
    if method == 'auto':
        method = 'pointbiserial' if is_binary else 'pearson'
    
    # Calculate correlation
    if method == 'pointbiserial':
        if not is_binary:
            print("Warning: Point-biserial requested but M1 is not binary. Using Pearson instead.")
            method = 'pearson'
        else:
            # Convert to int for pointbiserialr (expects binary as 0/1 integers)
            m1_binary = [int(x) for x in m1_values]
            corr, p_value = stats.pointbiserialr(m1_binary, m2_values)
    
    if method == 'pearson':
        corr, p_value = stats.pearsonr(m1_values, m2_values)
    
    return {
        'correlation': float(corr),
        'p_value': float(p_value),
        'n_samples': len(m1_values),
        'method': method,
        'is_binary_m1': is_binary,
        'interpretation': interpret_correlation(corr, p_value)
    }


def interpret_correlation(corr: float, p_value: float) -> str:
    """Provide human-readable interpretation of correlation."""
    # Significance
    if p_value < 0.001:
        sig = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        sig = "very significant (p < 0.01)"
    elif p_value < 0.05:
        sig = "significant (p < 0.05)"
    else:
        sig = "not significant (p >= 0.05)"
    
    # Strength
    abs_corr = abs(corr)
    if abs_corr < 0.1:
        strength = "negligible"
    elif abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.5:
        strength = "moderate"
    elif abs_corr < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    # Direction
    direction = "positive" if corr > 0 else "negative"
    
    return f"{strength} {direction} correlation, {sig}"


def main():
    parser = argparse.ArgumentParser(
        description='M3: Calculate correlation between M1 metric and M2 LLJ distance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # For MCQ/TF questions (binary flip vs LLJ distance) - uses point-biserial:
    python m3_pcc.py m2_output.jsonl
    python m3_pcc.py m2_output.jsonl --method pointbiserial
    
    # For report generation (BERTScore similarity vs LLJ distance) - uses Pearson:
    python m3_pcc.py combined.jsonl --m1-field bertscore_similarity --method pearson
    
    # Auto-detect method based on M1 values:
    python m3_pcc.py data.jsonl --method auto
    
Methods:
    - auto: Auto-detect (pointbiserial for binary M1, pearson for continuous)
    - pointbiserial: Point-biserial correlation (binary M1 vs continuous M2)
    - pearson: Pearson correlation (continuous M1 vs continuous M2)
        """
    )
    parser.add_argument('input_file', type=str, help='Path to JSONL with M1 and M2 fields')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path for output JSON results file')
    parser.add_argument('--m1-field', '--flipped-field', type=str, default='flipped',
                        dest='flipped_field',
                        help='Field name for M1 metric: flipped (MCQ) or bertscore_similarity (reports)')
    parser.add_argument('--m2-field', '--distance-field', type=str, default='distance',
                        dest='distance_field',
                        help='Field name for M2 LLJ distance (default: distance)')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'pearson', 'pointbiserial'],
                        help='Correlation method (default: auto)')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n📊 M3 Correlation Calculator")
    print("=" * 60)
    print(f"Input file:     {args.input_file}")
    print(f"M1 field:       {args.flipped_field}")
    print(f"M2 field:       {args.distance_field}")
    print(f"Method:         {args.method}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} entries")
    
    # Extract values
    m1_values, m2_values, skipped = extract_pairs(
        data,
        flipped_field=args.flipped_field,
        distance_field=args.distance_field
    )
    
    print(f"Valid pairs: {len(m1_values)}")
    print(f"Skipped (missing values): {skipped}")
    
    # Calculate correlation
    print(f"\nCalculating correlation (method: {args.method})...")
    results = calculate_correlation(m1_values, m2_values, method=args.method)
    
    # Add metadata
    results['input_file'] = str(args.input_file)
    results['m1_field'] = args.flipped_field
    results['m2_field'] = args.distance_field
    
    # Add summary statistics for M1
    if m1_values:
        unique_m1 = set(m1_values)
        if unique_m1 <= {0.0, 1.0}:
            # Binary M1 - show flip rate
            results['m1_mean'] = sum(m1_values) / len(m1_values)
            results['m1_sum'] = sum(m1_values)
        else:
            # Continuous M1 - show mean/std
            results['m1_mean'] = float(np.mean(m1_values))
            results['m1_std'] = float(np.std(m1_values))
    
    if m2_values:
        results['m2_mean'] = float(np.mean(m2_values))
        results['m2_std'] = float(np.std(m2_values))
        results['m2_min'] = float(min(m2_values))
        results['m2_max'] = float(max(m2_values))
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 RESULTS")
    print("=" * 60)
    
    if results.get('error'):
        print(f"⚠️  Error: {results['error']}")
    else:
        method_name = results.get('method', 'unknown').replace('pointbiserial', 'Point-Biserial').replace('pearson', 'Pearson')
        print(f"{method_name} Correlation (r): {results['correlation']:+.4f}")
        print(f"P-value:                   {results['p_value']:.4e}")
        print(f"Sample size (n):           {results['n_samples']}")
        if results.get('is_binary_m1'):
            print(f"M1 type:                   Binary (0/1)")
        else:
            print(f"M1 type:                   Continuous")
        print("-" * 60)
        print(f"Interpretation: {results['interpretation']}")
    
    print("-" * 60)
    if 'm1_sum' in results:
        # Binary M1
        print(f"M1 ({args.flipped_field}): {results['m1_sum']:.0f}/{results['n_samples']} = {results['m1_mean']:.4f}")
    elif 'm1_mean' in results:
        # Continuous M1
        print(f"M1 ({args.flipped_field}): mean={results['m1_mean']:+.4f}, std={results.get('m1_std', 0):.4f}")
    if 'm2_mean' in results:
        print(f"M2 ({args.distance_field}): mean={results['m2_mean']:+.4f}, std={results['m2_std']:.4f}")
    print("=" * 60)
    
    # Save results if output specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")
    else:
        # Default output path
        output_path = Path(args.input_file).stem + "_pcc.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
