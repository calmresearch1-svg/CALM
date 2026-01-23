#!/usr/bin/env python3
"""
JSONL Utility Script - Sort and Analyze

Sorts entries by entry_id and identifies missing entries.

Usage:
    python sort_jsonl.py input.jsonl [--output sorted.jsonl] [--inplace]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_jsonl(entries: List[Dict[str, Any]], file_path: str):
    """Save entries to JSONL file."""
    with open(file_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')


def sort_and_analyze(input_file: str, output_file: str = None, inplace: bool = False):
    """Sort entries by entry_id and find missing ones."""
    print(f"\n📂 Loading: {input_file}")
    entries = load_jsonl(input_file)
    print(f"   Total entries: {len(entries)}")
    
    # Extract entry_ids
    entry_ids = []
    for e in entries:
        eid = e.get('entry_id')
        if eid is not None:
            try:
                entry_ids.append(int(eid))
            except ValueError:
                entry_ids.append(eid)  # Keep as string if not numeric
    
    # Check if all numeric
    all_numeric = all(isinstance(eid, int) for eid in entry_ids)
    
    if all_numeric and entry_ids:
        # Sort entries by numeric entry_id
        sorted_entries = sorted(entries, key=lambda x: int(x.get('entry_id', 0)))
        
        # Find missing entry_ids
        min_id = min(entry_ids)
        max_id = max(entry_ids)
        all_expected = set(range(min_id, max_id + 1))
        present = set(entry_ids)
        missing = sorted(all_expected - present)
        
        print(f"\n📊 Analysis:")
        print(f"   Entry ID range: {min_id} → {max_id}")
        print(f"   Expected entries: {len(all_expected)}")
        print(f"   Present entries: {len(present)}")
        
        if missing:
            print(f"\n⚠️  Missing {len(missing)} entries:")
            if len(missing) <= 20:
                print(f"   {missing}")
            else:
                print(f"   First 10: {missing[:10]}")
                print(f"   Last 10: {missing[-10:]}")
        else:
            print(f"\n✅ No missing entries!")
        
        # Check for duplicates
        if len(entry_ids) != len(present):
            duplicates = len(entry_ids) - len(present)
            print(f"\n⚠️  Found {duplicates} duplicate entry_ids")
    else:
        # Sort as strings
        sorted_entries = sorted(entries, key=lambda x: str(x.get('entry_id', '')))
        print(f"\n⚠️  Non-numeric entry_ids detected, sorted alphabetically")
    
    # Determine output path
    if inplace:
        out_path = input_file
    elif output_file:
        out_path = output_file
    else:
        p = Path(input_file)
        out_path = str(p.parent / f"{p.stem}_sorted{p.suffix}")
    
    # Save sorted file
    save_jsonl(sorted_entries, out_path)
    print(f"\n💾 Saved sorted file: {out_path}")
    
    return missing if all_numeric else []


def main():
    parser = argparse.ArgumentParser(description="Sort JSONL by entry_id and find missing entries")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--inplace", "-i", action="store_true", help="Sort in place")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    sort_and_analyze(args.input, args.output, args.inplace)


if __name__ == "__main__":
    main()
