#!/usr/bin/env python3
"""
Calculate mean and std from JSONL scores, deduplicating by entry_id (first occurrence).

Usage:
    python calc_stats.py input.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
import statistics


def calc_stats(file_path: str):
    """Calculate mean and std, keeping first occurrence of duplicates."""
    seen_ids = set()
    scores = []
    duplicates = 0
    nulls = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry_id = entry.get('entry_id')
            
            # Skip duplicates
            if entry_id in seen_ids:
                duplicates += 1
                continue
            seen_ids.add(entry_id)
            
            # Get score
            score = entry.get('llm_judge_score')
            if score is not None:
                scores.append(score)
            else:
                nulls += 1
    
    print(f"\n📊 Statistics for: {Path(file_path).name}")
    print(f"   Total unique entries: {len(seen_ids)}")
    print(f"   Valid scores: {len(scores)}")
    print(f"   Null scores: {nulls}")
    print(f"   Duplicates skipped: {duplicates}")
    
    if scores:
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0
        print(f"\n   Mean: {mean:.4f}")
        print(f"   Std:  {std:.4f}")
    else:
        print("\n   ⚠️ No valid scores found")


def main():
    parser = argparse.ArgumentParser(description="Calculate mean/std from JSONL scores")
    parser.add_argument("input", nargs='+', help="Input JSONL file(s)")
    args = parser.parse_args()
    
    for f in args.input:
        if Path(f).exists():
            calc_stats(f)
        else:
            print(f"Error: File not found: {f}")


if __name__ == "__main__":
    main()
