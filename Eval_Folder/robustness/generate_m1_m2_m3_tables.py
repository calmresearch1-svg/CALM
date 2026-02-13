import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import glob
import re
import argparse
import sys

def parse_p_value(p):
    if p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.3f}"

def calculate_correlations(input_dir, llj_dir, output_dir, dataset_filter=None):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Scanning {input_dir} for metrics files...")
    metrics_files = glob.glob(os.path.join(input_dir, "*_flip_metrics.json"))
    
    tasks = [] # List of (dataset, model, metric_file, diff_file)
    
    for mf in metrics_files:
        basename = os.path.basename(mf)
        # Assuming format: {dataset}_{model}_flip_metrics.json
        prefix = basename.replace("_flip_metrics.json", "")
        
        dataset = None
        model = None
        
        known_datasets = ["iu_xray", "ham10000"]
        for d in known_datasets:
            if prefix.startswith(d + "_"):
                dataset = d
                model = prefix[len(d)+1:] # Rest is model
                break
        
        if not dataset:
            print(f"Warning: Could not parse dataset from {basename}. Skipping.")
            continue
            
        if dataset_filter and dataset != dataset_filter:
            continue
            
        diff_file = os.path.join(input_dir, f"{dataset}_{model}_diff_log.jsonl")
        
        if not os.path.exists(diff_file):
            print(f"Warning: Diff file not found for {basename}. Expected: {diff_file}")
            continue
            
        tasks.append({
            "dataset": dataset,
            "model": model,
            "metric_path": mf,
            "diff_path": diff_file
        })
    
    print(f"Found {len(tasks)} valid tasks to process.")

    rows_m1 = []
    rows_m2 = []
    rows_m3 = []
    rows_add = []
    
    matches_m2 = {} 
    
    # Judges to look for
    judges_map = {
        "gemini": ["gemini-2-5-flash", "gemini-2.5-flash"],
        "gpt": ["gpt-4-1-2025-04-14", "gpt-4"],
        "llama": ["meta-llama-3-2-90b-vision-instruct", "meta-llama-3-70b"]
    }
    
    # PROCESS TASKS
    for task in tasks:
        dataset = task['dataset']
        model = task['model']
        
        print(f"Processing {dataset} - {model}...")
        
        # --- M1 Processing ---
        with open(task['metric_path'], 'r') as f:
            m1_data = json.load(f)
            
        rows_m1.append({
            "Dataset": dataset,
            "Model": model,
            "Flip Rate": round(m1_data.get('flip_rate', 0), 4),
            "To Abstain Ratio": round(m1_data.get('m1_metrics', {}).get('m1_2_abstention_change_ratio', 0), 4),
            "Both Correct Ratio": round(m1_data.get('m1_metrics', {}).get('m1_3_correctness_consistency', 0), 4),
        })
        
        # --- Loading Diff Log ---
        diffs = []
        with open(task['diff_path'], 'r') as f:
            for line in f:
                if line.strip():
                    diffs.append(json.loads(line))
        df_diff = pd.DataFrame(diffs)
        
        # --- M2 & M3 Processing ---
        
        m2_entry = {"Dataset": dataset, "Model": model}
        
        search_path_1 = os.path.join(llj_dir, dataset)
        search_path_2 = llj_dir
        
        # Collect candidates
        candidates = []
        if os.path.isdir(search_path_1):
            candidates.extend(glob.glob(os.path.join(search_path_1, "*.jsonl")))
        if os.path.isdir(search_path_2):
             candidates.extend(glob.glob(os.path.join(search_path_2, "*.jsonl")))
             
        for judge_key, judge_patterns in judges_map.items():
            # Find matching LLJ file
            llj_file = None
            for c in candidates:
                bname = os.path.basename(c).lower()
                if model.lower() in bname: 
                    if any(p in bname for p in judge_patterns):
                        llj_file = c
                        break
            
            judge_col_name = f"{judge_key.capitalize()} Score"
            
            if not llj_file:
                print(f"  [Warn] No LLJ file found for {model} + {judge_key}")
                m2_entry[judge_col_name] = "N/A"
                continue
                
            # Load Scores
            scores_map = {}
            with open(llj_file, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    entry = json.loads(line)
                    
                    # ID Parsing
                    qid = entry.get('question_index')
                    if qid is None:
                        eid = entry.get("entry_id")
                        if eid is not None:
                             # Try integer conversion if possible
                            try:
                                qid = int(eid)
                            except ValueError:
                                qid = eid
                        else:
                            qid = entry.get('question_id')
                            
                    # Score Parsing
                    rating = entry.get('score')
                    if rating is None: rating = entry.get('llm_judge_score')
                    
                    # Fallback regex parsing if needed
                    if rating is None:
                        text = str(entry.get('judgment', ''))
                        m = re.search(r'\[\[(\d+)\]\]', text)
                        if m: rating = int(m.group(1))
                        else:
                            m2 = re.match(r'^\s*(\d+)', text)
                            if m2: rating = int(m2.group(1))

                    if rating is not None and qid is not None:
                        scores_map[qid] = rating
            
            df_diff['score'] = df_diff['question_id'].map(scores_map)
            
            df_scored = df_diff.dropna(subset=['score'])
            
            df_scored.loc[:, 'score_norm'] = (df_scored['score'] - 1) / 4.0
            
            # --- M2 Calculation ---
            if len(df_scored) > 0:
                mean_s = df_scored['score_norm'].mean()
                std_s = df_scored['score_norm'].std()
                m2_entry[judge_col_name] = f"{mean_s:.4f} ± {std_s:.4f}"
            else:
                m2_entry[judge_col_name] = "N/A"
                
            # --- M3 Calculation ---
            # 1. Flip vs Score
            if len(df_scored) > 5 and df_scored['is_flip'].nunique() > 1:
                r, p = stats.pointbiserialr(df_scored['is_flip'], df_scored['score'])
                corr_flip = f"{r:.3f} (p={parse_p_value(p)})"
                if p < 0.05: corr_flip = f"**{corr_flip}**"
            else:
                 corr_flip = "N/A"
                 
            # 2. To Abstain vs Score (Subset: Flips)
            df_flips = df_scored[df_scored['is_flip'] == True].copy()
            df_flips['is_to_abstain'] = (df_flips['counterfactual_extracted_answer'] == 'abstain')
            
            if len(df_flips) > 5 and df_flips['is_to_abstain'].nunique() > 1:
                r, p = stats.pointbiserialr(df_flips['is_to_abstain'], df_flips['score'])
                corr_abstain = f"{r:.3f} (p={parse_p_value(p)})"
                if p < 0.05: corr_abstain = f"**{corr_abstain}**"
            else:
                corr_abstain = "N/A"
                
            # 3. Both Correct vs Score (Subset: Both Answered)
            both_answered_indices = []
            both_correct_flags = []
            
            for idx, row in df_scored.iterrows():
                o_ans = str(row.get('original_extracted_answer', '')).lower()
                c_ans = str(row.get('counterfactual_extracted_answer', '')).lower()
                
                gt = str(row.get('ground_truth', '')).lower().strip()
                
                # Both Answered
                if o_ans != 'abstain' and c_ans != 'abstain':
                    both_answered_indices.append(idx)
                    
                    # Both Correct (Robust Logic)
                    def normalize_gt_for_m3(val):
                        if not val: return None
                        s = str(val).lower().strip().rstrip('.')
                        if s == 'yes' or s.startswith('yes'): return 'yes'
                        if s == 'no' or s.startswith('no'): return 'no'
                        # HAM10000 "C:hand" -> "c"
                        if ':' in s: 
                            parts = s.split(':')
                            if len(parts[0]) == 1: return parts[0]
                        if len(s) == 1 and s.isalpha(): return s
                        return None

                    gt_norm = normalize_gt_for_m3(row.get('ground_truth', ''))
                    
                    is_o = (o_ans == gt_norm)
                    is_c = (c_ans == gt_norm)
                    
                    both_correct_flags.append(1 if (is_o and is_c) else 0)
            
            df_subset = df_scored.loc[both_answered_indices].copy()
            df_subset['is_correct_metric'] = both_correct_flags
            
            if len(df_subset) > 5 and df_subset['is_correct_metric'].nunique() > 1:
                r, p = stats.pointbiserialr(df_subset['is_correct_metric'], df_subset['score'])
                corr_correct = f"{r:.3f} (p={parse_p_value(p)})"
                if p < 0.05: corr_correct = f"**{corr_correct}**"
            else:
                corr_correct = "N/A"

            rows_m3.append({
                "Dataset": dataset,
                "Model": model,
                "Judge": judge_key,
                "Corr(Flip vs Unchanged)": corr_flip,
                "Corr(To Abstain vs Other Flips)": corr_abstain,
                "Corr(Both Correct vs Other Answered)": corr_correct
            })
        
        rows_m2.append(m2_entry)
        
        # --- Additional Stats ---
        add_metrics = m1_data.get('additional_metrics', {})
        
        # Simple counts
        n_both_answered = len(df_diff[(df_diff['original_extracted_answer'] != 'abstain') & (df_diff['counterfactual_extracted_answer'] != 'abstain')])
        n_one_abstain = len(df_diff[(df_diff['original_extracted_answer'] == 'abstain') | (df_diff['counterfactual_extracted_answer'] == 'abstain')])
        
        rows_add.append({
            "Dataset": dataset,
            "Model": model,
            "Orig Accuracy (vs GT)": add_metrics.get('original_accuracy_gt', 'N/A'),
            "CF Accuracy (vs GT)": add_metrics.get('cf_accuracy_gt', 'N/A'),
            "Orig Abstention Rate": add_metrics.get('original_abstention_rate', 'N/A'),
            "CF Abstention Rate": add_metrics.get('cf_abstention_rate', 'N/A'),
            "Both Answered Count": n_both_answered,
            "At Least 1 Abstain Count": n_one_abstain
        })


    # OUTPUT
    print(f"\nExample M1 Rows: {len(rows_m1)}")
    if rows_m1:
        pd.DataFrame(rows_m1).to_csv(os.path.join(output_dir, "m1_metrics.csv"), index=False)
        
    print(f"Example M2 Rows: {len(rows_m2)}")
    if rows_m2:
        pd.DataFrame(rows_m2).to_csv(os.path.join(output_dir, "m2_scores.csv"), index=False)
        
    print(f"Example M3 Rows: {len(rows_m3)}")
    if rows_m3:
        pd.DataFrame(rows_m3).to_csv(os.path.join(output_dir, "m3_correlations.csv"), index=False)
        
    if rows_add:
        pd.DataFrame(rows_add).to_csv(os.path.join(output_dir, "additional_stats.csv"), index=False)
        
    print(f"Done. Tables saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate M1, M2, M3 tables for robustness submission")
    parser.add_argument("--input_dir", required=True, help="Directory containing *_flip_metrics.json and *_diff_log.jsonl")
    parser.add_argument("--llj_dir", required=True, help="Directory containing LLJ output JSONL files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output CSVs")
    parser.add_argument("--dataset", required=False, help="Filter by dataset name (iu_xray, ham10000)")
    
    args = parser.parse_args()
    
    calculate_correlations(args.input_dir, args.llj_dir, args.output_dir, args.dataset)
