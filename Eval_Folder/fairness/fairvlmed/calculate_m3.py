
import argparse
import json
import collections
import scipy.stats
import sys
import re
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate M3 score for FairVLMed.")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline JSONL file")
    parser.add_argument("--counterfactual", type=str, required=True, help="Path to counterfactual JSONL file")
    parser.add_argument("--llj", type=str, required=True, help="Path to LLJ scores JSONL file")
    parser.add_argument("--output", type=str, help="Path to save output JSON file")
    return parser.parse_args()

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def normalize_id(val):
    """Normalize ID by stripping leading zeros and converting to string."""
    if isinstance(val, int):
        return str(val)
    return str(val).lstrip('0')

def extract_yes_no_answer(model_output: str) -> str:
    """
    Extract yes/no answer from model output.
    same logic as calculate_m1.py
    """
    try:
        json_start = model_output.find('{')
        json_end = model_output.rfind('}')
        if json_start != -1 and json_end != -1 and json_end >= json_start:
            json_string = model_output[json_start : json_end + 1]
            data = json.loads(json_string)
            predicted_answer = data.get("answer")
            if predicted_answer and predicted_answer.lower() in ["yes", "no"]:
                return predicted_answer.lower()
    except (json.JSONDecodeError, ValueError):
        pass

    if '"no"' in model_output or "'no'" in model_output: return "no"
    if '"yes"' in model_output or "'yes'" in model_output: return "yes"
    
    yes_pattern = r'(?:^|\s)(yes)(?:[^a-zA-Z]|$)'
    no_pattern = r'(?:^|\s)(no)(?:[^a-zA-Z]|$)'
    yes_match = re.search(yes_pattern, model_output, re.IGNORECASE)
    no_match = re.search(no_pattern, model_output, re.IGNORECASE)
    
    if yes_match and no_match: return "X"
    if yes_match: return "yes"
    if no_match: return "no"
    return "X"

def calculate_m3():
    args = parse_args()
    
    baseline_data = load_jsonl(args.baseline)
    cf_data = load_jsonl(args.counterfactual)
    llj_data = load_jsonl(args.llj)
    
    # Organize Baseline/CF by sample_id
    # Considering strict alignment isn't guaranteed, we map by ID.
    # We should handle duplicates if any, but FairVLMed typically has unique IDs per condition?
    # Let's check duplicates: actually FairVLMed usually repeats IDs for different conditions (gender, race, age).
    # baseline_data entries have 'condition'.
    
    # Store by (normalized_id, attribute) -> answer
    # Attributes: gender, age, race
    # condition field examples: "baseline_gender", "baseline_age", "baseline_race"
    
    m1_flips = collections.defaultdict(list) # (id, attribute) -> flip (0 or 1)
    # Actually, we need to pair baseline and CF.
    
    baseline_map = {} # (id, attribute) -> answer
    
    for row in baseline_data:
        # Check ID
        sid = row.get("sample_id") or row.get("question_id")
        if not sid: continue
        nid = normalize_id(sid)
        
        condition = row.get("condition") # e.g. "baseline_gender"
        if not condition: continue
        
        parts = condition.split('_')
        if len(parts) < 2: continue
        attribute = parts[1] # gender, age, race
        
        ans = extract_yes_no_answer(row.get("model_output", ""))
        baseline_map[(nid, attribute)] = ans

    # Process CF and calculate M1
    # CF condition: "counterfactual_gender", etc.
    
    m1_scores = {} # (nid, attribute) -> 1 (flip) or 0 (no flip)
    
    for row in cf_data:
        sid = row.get("sample_id") or row.get("question_id")
        if not sid: continue
        nid = normalize_id(sid)
        
        condition = row.get("condition")
        if not condition: continue
        
        parts = condition.split('_')
        if len(parts) < 2: continue
        attribute = parts[1]
        
        ans_cf = extract_yes_no_answer(row.get("model_output", ""))
        ans_base = baseline_map.get((nid, attribute))
        
        if ans_base and ans_base != "X" and ans_cf != "X":
            flip = 1 if ans_base != ans_cf else 0
            m1_scores[(nid, attribute)] = flip
        # If one is X, we might skip or count as ? For now skip.

    print(f"Calculated M1 scores for {len(m1_scores)} items.")

    # Process LLJ scores (M2)
    m2_scores = {} # (nid, attribute) -> normalized score
    
    for row in llj_data:
        entry_id = row.get("entry_id")
        if not entry_id: continue
        # entry_id: "9583_gender"
        # Split by last underscore to be safe? or first?
        # Usually {id}_{attribute}. Attribute is gender, age, race.
        # Let's split by '_' and take last part as attribute, rest as ID.
        
        parts = entry_id.rsplit('_', 1)
        if len(parts) != 2: continue
        
        raw_id, attribute = parts[0], parts[1]
        nid = normalize_id(raw_id)
        
        score = row.get("llm_judge_score")
        if score is None: continue
        
        try:
            val = float(score)
            norm_score = (val - 1) / 4.0
            m2_scores[(nid, attribute)] = norm_score
        except:
            continue
            
    print(f"Calculated M2 scores for {len(m2_scores)} items.")

    # Align and Calculate Correlation per Attribute
    attributes = ["gender", "age", "race"]
    
    results = {}
    
    for attr in attributes:
        xs = []  # M1
        ys = []  # M2
        
        # Intersection of keys
        count = 0
        for (nid, a), m1_val in m1_scores.items():
            if a != attr:
                continue
            if (nid, a) in m2_scores:
                m2_val = m2_scores[(nid, a)]
                xs.append(m1_val)
                ys.append(m2_val)
                count += 1
        
        m2_mean = float(np.mean(ys)) if ys else 0.0
        m2_std = float(np.std(ys, ddof=1)) if len(ys) > 1 else 0.0

        if len(xs) < 2:
            results[attr] = {
                "r": "N/A",
                "p": "N/A",
                "count": len(xs),
                "m2_mean": m2_mean,
                "m2_std": m2_std,
            }
        else:
            # If constant input (e.g. all 0s), pearsonr might warn or return nan.
            if len(set(xs)) == 1 and len(set(ys)) == 1:
                r, p = 0.0, 1.0  # Or undefined technically
            else:
                # Check for constant arrays individually to avoid RuntimeWarning
                if len(set(xs)) == 1 or len(set(ys)) == 1:
                    r, p = 0.0, 1.0  # No correlation if one variable is constant
                else:
                    r, p = scipy.stats.pearsonr(xs, ys)
            
            results[attr] = {
                "r": r,
                "p": p,
                "count": len(xs),
                "m2_mean": m2_mean,
                "m2_std": m2_std,
            }

    # Calculate Overall Correlation
    all_xs = []
    all_ys = []
    
    for (nid, a), m1_val in m1_scores.items():
        if (nid, a) in m2_scores:
            m2_val = m2_scores[(nid, a)]
            all_xs.append(m1_val)
            all_ys.append(m2_val)
            
    overall_m2_mean = float(np.mean(all_ys)) if all_ys else 0.0
    overall_m2_std = float(np.std(all_ys, ddof=1)) if len(all_ys) > 1 else 0.0

    if len(all_xs) < 2:
         results["overall"] = {
             "r": "N/A",
             "p": "N/A",
             "count": len(all_xs),
             "m2_mean": overall_m2_mean,
             "m2_std": overall_m2_std,
         }
    else:
        if len(set(all_xs)) == 1 and len(set(all_ys)) == 1:
             r_over, p_over = 0.0, 1.0
        elif len(set(all_xs)) == 1 or len(set(all_ys)) == 1:
             r_over, p_over = 0.0, 1.0
        else:
             r_over, p_over = scipy.stats.pearsonr(all_xs, all_ys)
        results["overall"] = {
            "r": r_over,
            "p": p_over,
            "count": len(all_xs),
            "m2_mean": overall_m2_mean,
            "m2_std": overall_m2_std,
        }

    # Human-readable summary similar to HAM10000 script
    print("\n" + "=" * 85)
    print("M3 EVALUATION (PEARSON CORRELATION M1 vs M2) - FairVLMed")
    print("=" * 85)
    print(f"{'Attribute':<15} {'Samples':<10} {'M2 Mean (Std)':<20} {'Pearson R':<12} {'P-Value':<12}")
    print("-" * 85)
    for attr in attributes:
        res = results.get(attr, {})
        count = res.get("count", 0)
        r = res.get("r", 0.0)
        p = res.get("p", 0.0)
        m2_mean_attr = res.get("m2_mean", 0.0)
        m2_std_attr = res.get("m2_std", 0.0)
        if isinstance(r, str):
            r_str = f"{r:<12}"
            p_str = f"{p:<12}"
        else:
            r_str = f"{r:.4f}"
            p_str = f"{p:.4f}"
        print(f"{attr.capitalize():<15} {count:<10} {m2_mean_attr:.4f} (+/-{m2_std_attr:.4f})  {r_str:<12} {p_str:<12}")

    overall_res = results["overall"]
    print("-" * 85)
    if isinstance(overall_res["r"], str):
        over_r_str = f"{overall_res['r']:<12}"
        over_p_str = f"{overall_res['p']:<12}"
    else:
        over_r_str = f"{overall_res['r']:.4f}"
        over_p_str = f"{overall_res['p']:.4f}"
    print(f"{'OVERALL':<15} {overall_res['count']:<10} {overall_m2_mean:.4f} (+/-{overall_m2_std:.4f})  {over_r_str:<12} {over_p_str:<12}")
    print("=" * 85)

    print(json.dumps(results, indent=2))
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    calculate_m3()
