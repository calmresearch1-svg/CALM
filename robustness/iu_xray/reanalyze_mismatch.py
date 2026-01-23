"""
reanalyze_mismatch_results.py

Reanalyzes saved results with improved mismatch detection.
Fixes the poor mismatch detection by using better heuristics based on actual model responses.

Usage:
    python reanalyze_mismatch_results.py <input_results.jsonl> <output_reanalyzed.jsonl>
"""

import json
import re
from pathlib import Path
from typing import Tuple, Dict
from collections import defaultdict
import argparse


class ImprovedAbstentionDetector:
    """Enhanced abstention and mismatch detection based on actual response patterns."""
    
    def __init__(self):
        # Combine your phrases with ours
        self.abstention_phrases = [
            # Your phrases
            "can't answer", "no answer", "cannot answer", "not appropriate",
            "refuse to answer", "sorry", "inappropriate", "cannot provide",
            "apologize", "not visible", 
            "can not be inferred from the image", "can not", "qualified", 
            "professional", "however", "can no", "patient", "clinical", "definitive",
            
            # Our additional phrases
            "i don't know", "i do not know", "cannot determine",
            "unable to assess", "cannot assess", "unclear",
            "i'm not sure", "not certain",
            "insufficient information",
            "can't tell", "cannot tell",
            "don't have enough", "do not have enough",
            "not possible to", "impossible to determine",
            "cannot be determined", "can't be determined",
        ]
        
        # CRITICAL: Mismatch detection patterns based on actual responses
        self.explicit_mismatch_patterns = [
            # Direct "not a X" patterns
            r'not an? (brain|abdominal|spine|knee|pelvic|chest)?\s*(ct scan|mri|ultrasound|x-ray)',
            r'not an? (ct scan|mri|ultrasound|x-ray)',
            
            # "is a X, not a Y" patterns
            r'is an? (chest )?x-ray,?\s*not',
            r'provided is an? (chest )?x-ray',
            
            # "this is X, not Y" variations
            r'this is an? .+,?\s*not an?',
            r'image is an? .+,?\s*not an?',
            
            # Explicit modality/region statements
            r'chest x-ray,?\s*not an? (brain|abdominal|spine|knee|pelvic)',
            r'x-ray.{0,20}not an? (ct|mri|ultrasound)',
        ]
        
        # Implicit mismatch indicators (weaker but still relevant)
        self.implicit_mismatch_phrases = [
            "based on this image",  # followed by "not possible"
            "from this image",
            "the image provided",
            "the question seems to be based on an incorrect image",
            "if you have a",  # "if you have a brain CT scan..."
            "please provide the correct image",
            "please provide the appropriate image",
        ]
    
    def detect_mismatch(self, response: str) -> Tuple[bool, str]:
        """
        Detect explicit mismatch recognition in response.
        Returns: (mismatch_detected, evidence)
        """
        response_lower = response.lower()
        
        # Check explicit patterns (high confidence)
        for pattern in self.explicit_mismatch_patterns:
            match = re.search(pattern, response_lower)
            if match:
                evidence = match.group(0)
                return True, f"explicit_pattern: {evidence}"
        
        # Check for "is X, not Y" structure
        if re.search(r'(is|provided is).+(not an?|not a)', response_lower):
            # Extract the relevant part
            match = re.search(r'(.{0,50}(is|provided is).{0,50}not.{0,50})', response_lower)
            if match:
                evidence = match.group(0).strip()
                return True, f"is_not_structure: {evidence[:80]}"
        
        return False, "no_mismatch"
    
    def detect_abstention(self, response: str) -> Tuple[bool, str]:
        """
        Detect abstention in response.
        Returns: (is_abstention, reason)
        """
        response_lower = response.lower().strip()
        
        # Check direct phrases
        for phrase in self.abstention_phrases:
            if phrase in response_lower:
                return True, f"direct_phrase: {phrase}"
        
        # Check "not possible to" pattern (very common in mismatch responses)
        if "not possible to" in response_lower or "impossible to" in response_lower:
            return True, "not_possible_pattern"
        
        # Check "cannot/can't" patterns
        if re.search(r"can'?not|can'?t", response_lower):
            # Make sure it's about answering/determining, not about the image
            if any(word in response_lower for word in ["answer", "determine", "assess", "tell", "provide"]):
                return True, "cannot_answer_pattern"
        
        # CRITICAL: Check for "therefore, it does not" pattern (partial abstention)
        # Model explains mismatch but still provides an answer
        if "therefore, it does not" in response_lower:
            return True, "partial_abstention_therefore"
        
        # Check "does not provide information" pattern
        if "does not provide information" in response_lower:
            return True, "no_information_pattern"
        
        # Check "does not contain information" pattern  
        if "does not contain information" in response_lower:
            return True, "no_information_pattern"
        
        # Check "does not show" + negative statement pattern
        # e.g., "does not show any signs of", "does not depict"
        if re.search(r"does not (show|depict|indicate|contain)", response_lower):
            # Make sure it's in the context of explaining mismatch
            if any(phrase in response_lower for phrase in ["not a", "rather a", "provided is", "image is"]):
                return True, "negative_statement_with_mismatch"
        
        # Check "the correct answer based on" pattern (often follows mismatch explanation)
        # This is key: model says "can't answer from THIS image, but here's an answer anyway"
        if "correct answer based on" in response_lower or "correct answer to" in response_lower:
            # If preceded by mismatch detection, this is partial abstention
            if any(pattern in response_lower for pattern in ["not a", "rather a", "provided is a", "not but"]):
                return True, "forced_answer_after_mismatch"
        
        return False, "no_abstention"


def extract_prediction(response: str) -> str:
    """
    Extract yes/no prediction from model response.
    Returns: 'yes', 'no', 'abstain', or 'unclear'
    
    Uses your proven extraction logic.
    """
    response_lower = response.lower()
    
    # First check for abstention
    detector = ImprovedAbstentionDetector()
    is_abstain, _ = detector.detect_abstention(response)
    if is_abstain:
        return 'abstain'
    
    # Look for explicit answers in brackets or at the end
    bracket_match = re.search(r'\[(yes|no)\]', response_lower)
    if bracket_match:
        return bracket_match.group(1)
    
    # Count occurrences of yes and no (your method)
    yes_count = len(re.findall(r'\byes\b', response_lower))
    no_count = len(re.findall(r'\bno\b', response_lower))
    
    if yes_count > no_count:
        return 'yes'
    elif no_count > yes_count:
        return 'no'
    
    return 'unclear'


def reanalyze_entry(entry: Dict) -> Dict:
    """
    Reanalyze a single result entry with improved detection.
    """
    detector = ImprovedAbstentionDetector()
    response = entry['response']
    
    # Re-detect mismatch (CRITICAL FIX)
    mismatch_detected, mismatch_evidence = detector.detect_mismatch(response)
    
    # Re-detect abstention
    is_abstention, abstention_reason = detector.detect_abstention(response)
    
    # Re-extract answer
    extracted_answer = extract_prediction(response)
    
    # NEW: Detect "partial abstention" - model explains mismatch but still answers
    # This is problematic behavior: model knows it's wrong but answers anyway
    partial_abstention = False
    if mismatch_detected and extracted_answer in ['yes', 'no']:
        # Model detected mismatch but still gave yes/no answer
        partial_abstention = True
        # Override: treat partial abstention as full abstention
        if not is_abstention:
            is_abstention = True
            abstention_reason = "partial_abstention_override"
    
    # Update entry
    entry['mismatch_detected'] = mismatch_detected
    entry['mismatch_evidence'] = mismatch_evidence
    entry['is_abstention'] = is_abstention
    entry['abstention_reason'] = abstention_reason
    entry['extracted_answer'] = extracted_answer
    entry['partial_abstention'] = partial_abstention  # NEW field
    
    # Add combined flag: explicit mismatch detection (ideal behavior)
    entry['explicit_mismatch_and_abstention'] = mismatch_detected and is_abstention
    
    # NEW: Flag for problematic behavior (detects mismatch but answers anyway)
    entry['problematic_partial_abstention'] = mismatch_detected and extracted_answer in ['yes', 'no']
    
    return entry


def reanalyze_file(input_path: str, output_path: str) -> Dict:
    """
    Reanalyze an entire results file.
    Returns statistics.
    """
    print(f"\n{'='*70}")
    print(f"REANALYZING: {input_path}")
    print('='*70)
    
    # Load results
    with open(input_path, 'r') as f:
        results = [json.loads(line) for line in f]
    
    print(f"Loaded {len(results)} entries")
    
    # Reanalyze each entry
    reanalyzed = []
    for entry in results:
        reanalyzed_entry = reanalyze_entry(entry)
        reanalyzed.append(reanalyzed_entry)
    
    # Save reanalyzed results
    with open(output_path, 'w') as f:
        for entry in reanalyzed:
            f.write(json.dumps(entry) + '\n')
    
    print(f"✓ Saved reanalyzed results to: {output_path}")
    
    # Calculate statistics
    stats = calculate_statistics(reanalyzed)
    
    return stats


def calculate_statistics(results: list) -> Dict:
    """Calculate comprehensive statistics."""
    total = len(results)
    
    stats = {
        'total': total,
        'abstentions': sum(1 for r in results if r['is_abstention']),
        'mismatch_detected': sum(1 for r in results if r['mismatch_detected']),
        'explicit_mismatch_and_abstention': sum(1 for r in results if r.get('explicit_mismatch_and_abstention', False)),
        'answered_yes': sum(1 for r in results if r['extracted_answer'] == 'yes'),
        'answered_no': sum(1 for r in results if r['extracted_answer'] == 'no'),
        'unclear': sum(1 for r in results if r['extracted_answer'] == 'unclear'),
    }
    
    # Calculate rates
    stats['abstention_rate'] = stats['abstentions'] / total * 100 if total > 0 else 0
    stats['mismatch_detection_rate'] = stats['mismatch_detected'] / total * 100 if total > 0 else 0
    stats['explicit_mismatch_rate'] = stats['explicit_mismatch_and_abstention'] / total * 100 if total > 0 else 0
    
    return stats


def print_statistics(stats: Dict, condition: str):
    """Print statistics in a formatted way."""
    print(f"\n{'='*70}")
    print(f"STATISTICS: {condition}")
    print('='*70)
    print(f"Total entries: {stats['total']}")
    print(f"\nDetection Metrics:")
    print(f"  Abstention Rate:           {stats['abstention_rate']:6.2f}% ({stats['abstentions']}/{stats['total']})")
    print(f"  Mismatch Detection Rate:   {stats['mismatch_detection_rate']:6.2f}% ({stats['mismatch_detected']}/{stats['total']})")
    print(f"  Explicit Mismatch+Abstain: {stats['explicit_mismatch_rate']:6.2f}% ({stats['explicit_mismatch_and_abstention']}/{stats['total']})")
    print(f"\nAnswer Distribution:")
    print(f"  Yes:     {stats['answered_yes']:5d} ({stats['answered_yes']/stats['total']*100:5.2f}%)")
    print(f"  No:      {stats['answered_no']:5d} ({stats['answered_no']/stats['total']*100:5.2f}%)")
    print(f"  Abstain: {stats['abstentions']:5d} ({stats['abstention_rate']:5.2f}%)")
    print(f"  Unclear: {stats['unclear']:5d} ({stats['unclear']/stats['total']*100:5.2f}%)")
    print('='*70)


def show_examples(input_path: str, n: int = 5):
    """Show examples of mismatch detection."""
    with open(input_path, 'r') as f:
        results = [json.loads(line) for line in f]
    
    print(f"\n{'='*70}")
    print(f"EXAMPLE MISMATCH DETECTIONS (first {n})")
    print('='*70)
    
    mismatch_examples = [r for r in results if r.get('mismatch_detected', False)][:n]
    
    for i, entry in enumerate(mismatch_examples, 1):
        print(f"\n[Example {i}]")
        print(f"Question: {entry['question'][:80]}...")
        print(f"Response: {entry['response'][:150]}...")
        print(f"Evidence: {entry.get('mismatch_evidence', 'N/A')}")
        print(f"Abstained: {entry['is_abstention']}")
        print(f"Extracted: {entry['extracted_answer']}")
        print("-" * 70)


def compare_old_vs_new(input_path: str):
    """Compare old detection vs new detection."""
    print(f"\n{'='*70}")
    print("COMPARISON: Old vs New Detection")
    print('='*70)
    
    # Load original file (before reanalysis)
    with open(input_path, 'r') as f:
        old_results = [json.loads(line) for line in f]
    
    # Reanalyze in memory
    detector = ImprovedAbstentionDetector()
    new_mismatch_count = 0
    old_mismatch_count = sum(1 for r in old_results if r.get('mismatch_detected', False))
    
    for entry in old_results:
        mismatch_detected, _ = detector.detect_mismatch(entry['response'])
        if mismatch_detected:
            new_mismatch_count += 1
    
    total = len(old_results)
    
    print(f"\nTotal entries: {total}")
    print(f"\nOld detection:")
    print(f"  Mismatch detected: {old_mismatch_count} ({old_mismatch_count/total*100:.2f}%)")
    print(f"\nNew detection:")
    print(f"  Mismatch detected: {new_mismatch_count} ({new_mismatch_count/total*100:.2f}%)")
    print(f"\nImprovement:")
    print(f"  Additional mismatches found: {new_mismatch_count - old_mismatch_count}")
    print(f"  Improvement: +{(new_mismatch_count - old_mismatch_count)/total*100:.2f}%")
    print('='*70)


def batch_reanalyze(input_dir: str, output_dir: str):
    """Reanalyze all result files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Find all jsonl files
    jsonl_files = list(input_path.glob("*_results.jsonl"))
    
    if not jsonl_files:
        print(f"No result files found in {input_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"BATCH REANALYSIS: {len(jsonl_files)} files")
    print('='*70)
    
    all_stats = {}
    
    for input_file in jsonl_files:
        output_file = output_path / f"{input_file.stem}_reanalyzed.jsonl"
        
        # Reanalyze
        stats = reanalyze_file(str(input_file), str(output_file))
        
        # Extract condition from filename
        condition = input_file.stem.replace('_results', '').split('_', 1)[-1]
        all_stats[condition] = stats
        
        # Print stats
        print_statistics(stats, condition)
    
    # Print comparative summary
    print_comparative_summary(all_stats)


def print_comparative_summary(all_stats: Dict):
    """Print comparison across all conditions."""
    print(f"\n{'='*70}")
    print("COMPARATIVE SUMMARY")
    print('='*70)
    print(f"\n{'Condition':<30} {'Abstention %':<15} {'Mismatch %':<15} {'Explicit %':<15}")
    print('-'*70)
    
    for condition in ['original', 'cf1_wrong_modality', 'cf2_wrong_region', 'cf3_both_wrong']:
        if condition in all_stats:
            s = all_stats[condition]
            print(f"{condition:<30} {s['abstention_rate']:>6.2f}%{'':<8} "
                  f"{s['mismatch_detection_rate']:>6.2f}%{'':<8} "
                  f"{s['explicit_mismatch_rate']:>6.2f}%")
    
    print('='*70)


def main():
    parser = argparse.ArgumentParser(description='Reanalyze mismatch detection results')
    parser.add_argument('input', help='Input results file or directory')
    parser.add_argument('--output', help='Output file or directory (default: add _reanalyzed suffix)')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    parser.add_argument('--compare', action='store_true', help='Show comparison of old vs new')
    parser.add_argument('--examples', type=int, default=5, help='Number of examples to show')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing
        output_dir = args.output or Path(args.input).parent / 'reanalyzed'
        batch_reanalyze(args.input, str(output_dir))
    else:
        # Single file processing
        input_path = args.input
        
        # Show comparison first
        if args.compare:
            compare_old_vs_new(input_path)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_reanalyzed.jsonl"
        
        # Reanalyze
        stats = reanalyze_file(input_path, str(output_path))
        
        # Print statistics
        condition = Path(input_path).stem.replace('_results', '')
        print_statistics(stats, condition)
        
        # Show examples
        show_examples(str(output_path), n=args.examples)


if __name__ == "__main__":
    main()
