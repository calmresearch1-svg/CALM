import json
import argparse
import os
import pandas as pd

def calculate_metrics(original_file, cf_file, output_dir, model_name, dataset):
    print(f"Loading {original_file}...")
    orig_data = []
    with open(original_file, 'r') as f:
        for line in f:
            if line.strip():
                orig_data.append(json.loads(line))
    
    print(f"Loading {cf_file}...")
    cf_data = []
    with open(cf_file, 'r') as f:
        for line in f:
            if line.strip():
                cf_data.append(json.loads(line))
                
    # Create maps
    try:
        orig_map = {x['question_id']: x for x in orig_data}
        cf_map = {x['question_id']: x for x in cf_data}
    except KeyError:
        print("KeyError: 'question_id' not found, trying 'question_index'")
        orig_map = {x['question_index']: x for x in orig_data}
        cf_map = {x['question_index']: x for x in cf_data}
    
    # Find common IDs
    common_ids = sorted(list(set(orig_map.keys()) & set(cf_map.keys())))
    print(f"Comparing {len(common_ids)} matched Q&A pairs...")
    
    flips = 0
    to_abstain = 0
    from_abstain = 0 
    answer_flip = 0
    
    both_correct = 0
    both_answered = 0
    
    diff_log = []
    
    for qid in common_ids:
        o = orig_map[qid]
        c = cf_map[qid]
        
        o_ans = str(o.get('extracted_answer', '')).lower()
        c_ans = str(c.get('extracted_answer', '')).lower()
        
        # Check if flipped
        is_flip = (o_ans != c_ans)
        
        # Classify flip types
        flip_type = "unchanged"
        if is_flip:
            flips += 1
            if c_ans == 'abstain':
                to_abstain += 1
                flip_type = "to_abstain"
            elif o_ans == 'abstain':
                from_abstain += 1
                flip_type = "from_abstain"
            else:
                answer_flip += 1
                flip_type = "answer_flip"

        
        if o_ans != 'abstain' and c_ans != 'abstain':
            both_answered += 1
        
        diff_entry = {
            "question_id": qid,
            "original_extracted_answer": o_ans,
            "counterfactual_extracted_answer": c_ans,
            "is_flip": is_flip,
            "flip_type": flip_type,
            "ground_truth": o.get('ground_truth', ''),
            "original_response": o.get('response', ''),
            "counterfactual_response": c.get('response', '')
        }
        diff_log.append(diff_entry)
        
    # Save diff log
    diff_path = os.path.join(output_dir, f"{dataset}_{model_name}_diff_log.jsonl")
    with open(diff_path, 'w') as f:
        for entry in diff_log:
            f.write(json.dumps(entry) + "\n")
            
    # Calculate detailed stats
    transitions = {
        "unchanged": 0,
        "no_to_abstain": 0,
        "yes_to_abstain": 0,
        "abstain_to_no": 0,
        "abstain_to_yes": 0,
        "yes_to_no": 0,
        "no_to_yes": 0,
        "other_flip": 0
    }
    
    abstention_stats = {
        "original_abstention_count": 0,
        "counterfactual_abstention_count": 0,
        "both_abstain_count": 0
    }
    
    for qid in common_ids:
        o = orig_map[qid]
        c = cf_map[qid]
        
        o_ans = str(o.get('extracted_answer', '')).lower()
        c_ans = str(c.get('extracted_answer', '')).lower()
        
        # Abstention Stats
        is_o_abstain = (o_ans == 'abstain')
        is_c_abstain = (c_ans == 'abstain')
        
        if is_o_abstain: abstention_stats["original_abstention_count"] += 1
        if is_c_abstain: abstention_stats["counterfactual_abstention_count"] += 1
        if is_o_abstain and is_c_abstain: abstention_stats["both_abstain_count"] += 1
        
        # Transitions
        if o_ans == c_ans:
            transitions["unchanged"] += 1
        else:
            # Flips
            key = f"{o_ans}_to_{c_ans}"
            if key in transitions:
                transitions[key] += 1
            else:
                transitions["other_flip"] += 1

    total = len(common_ids)
    
    # Calculate Metrics
    flip_rate = flips / total if total > 0 else 0
    
    # M1.2: Abstention Change Ratio (How many flips are TO abstain?) 
    m1_2 = to_abstain / flips if flips > 0 else 0.0
    
    # M1.3 Correctness Consistency (Both Correct)
    both_correct_count = 0
    
    def normalize_gt(gt_in, full_entry=None):
        if full_entry and 'parsed_gt' in full_entry and full_entry['parsed_gt']:
             return str(full_entry['parsed_gt']).lower()

        if not gt_in: return None
        s = str(gt_in).lower().strip().rstrip('.')
        
        # Explicit Yes/No
        # Heuristic for IU X-Ray 
        if s in ['yes', 'no', 'abstain']: return s
        if s.startswith('yes'): return 'yes'
        if s.startswith('no'): return 'no'
        
        # Heuristic for HAM10000 where GT might be "C:hand"
        if ':' in s:
            parts = s.split(':')
            if len(parts[0]) == 1:
                return parts[0]
                
        # If single letter A-E
        if len(s) == 1 and s.isalpha():
            return s
            
        return None 

    for qid in common_ids:
        o = orig_map[qid]
        c = cf_map[qid]
        
        o_ans = str(o.get('extracted_answer', '')).lower()
        c_ans = str(c.get('extracted_answer', '')).lower()
        
        o_gt = normalize_gt(o.get('ground_truth', ''), o)
        
        is_o_correct = (o_ans == o_gt)
        
        is_c_correct = (c_ans == o_gt)
        
        if is_o_correct and is_c_correct:
            both_correct_count += 1

    # Additional Metrics 
    # 1. Original response GT match
    # 2. CF response GT match
    # 3. Original question abstention rate
    # 4. CF Question abstention rate
    
    orig_match_gt_count = 0
    cf_match_gt_count = 0
    orig_abstain_count = 0
    cf_abstain_count = 0
    
    # Answer Distribution Stats
    original_distribution = {}
    counterfactual_distribution = {}

    for qid in common_ids:
        o = orig_map[qid]
        c = cf_map[qid]
        
        o_ans = str(o.get('extracted_answer', '')).lower()
        c_ans = str(c.get('extracted_answer', '')).lower()
        
        original_distribution[o_ans] = original_distribution.get(o_ans, 0) + 1
        counterfactual_distribution[c_ans] = counterfactual_distribution.get(c_ans, 0) + 1
        
        o_gt = normalize_gt(o.get('ground_truth', ''), o)
        c_gt = normalize_gt(c.get('ground_truth', ''), c) # Should be same as o_gt usually
        
        # 1 & 2 GT Match
        if o_ans == o_gt: orig_match_gt_count += 1
        if c_ans == c_gt: cf_match_gt_count += 1
        
        # 3 & 4 Abstention
        if o_ans == 'abstain': orig_abstain_count += 1
        if c_ans == 'abstain': cf_abstain_count += 1

    m1_3 = both_correct_count / total if total > 0 else 0.0
    
    metrics = {
      "dataset": dataset,
      "total_pairs": total,
      "flip_rate": round(flip_rate, 4),
      "total_flips": flips,
      "transitions": transitions,
      "answer_distributions": {
          "original": original_distribution,
          "counterfactual": counterfactual_distribution
      },
      "abstention_stats": {
        "original_abstention_count": abstention_stats["original_abstention_count"],
        "original_abstention_rate": round(abstention_stats["original_abstention_count"] / total, 4) if total > 0 else 0,
        "counterfactual_abstention_count": abstention_stats["counterfactual_abstention_count"],
        "counterfactual_abstention_rate": round(abstention_stats["counterfactual_abstention_count"] / total, 4) if total > 0 else 0,
        "abstention_delta": round((abstention_stats["counterfactual_abstention_count"] - abstention_stats["original_abstention_count"]) / total, 4) if total > 0 else 0,
        "both_abstain_count": abstention_stats["both_abstain_count"]
      },
      "additional_metrics": {
          "original_accuracy_gt": round(orig_match_gt_count / total, 4) if total > 0 else 0,
          "cf_accuracy_gt": round(cf_match_gt_count / total, 4) if total > 0 else 0,
          "original_abstention_rate": round(orig_abstain_count / total, 4) if total > 0 else 0,
          "cf_abstention_rate": round(cf_abstain_count / total, 4) if total > 0 else 0
      },
      "m1_metrics": {
        "m1_1_flip_rate": round(flip_rate, 4),
        "m1_2_abstention_change_ratio": round(m1_2, 4),
        "m1_3_correctness_consistency": round(m1_3, 4)
      }
    }
    
    metrics_path = os.path.join(output_dir, f"{dataset}_{model_name}_flip_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"--- Results ---")
    print(f"Flip Rate: {metrics['flip_rate']:.2%}")
    print(f"Total Flips: {flips} / {total}")
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--counterfactual", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset", required=True)
    
    args = parser.parse_args()
    calculate_metrics(args.original, args.counterfactual, args.output_dir, args.model_name, args.dataset)
