import json
import argparse
import re
import sys
import os
import glob

# ==========================================
# LOGIC FOR IU X-Ray
# ==========================================

def iu_parse_json_response(response_str):
    """
    Attempts to parse a JSON-like string from the model response.
    Handles common issues like single quotes, markdown blocks, etc.
    """
    if not isinstance(response_str, str):
        return None

    # Cleaning: Remove markdown code blocks
    cleaned = re.sub(r'```json\s*', '', response_str)
    cleaned = re.sub(r'```', '', cleaned)
    cleaned = cleaned.strip()

    # Attempt 1: Standard JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Relaxed JSON (single quotes to double quotes)
    try:
        # Simple heuristic: if it looks like a python dict with single quotes
        if "'" in cleaned and '"' not in cleaned:
             import ast
             return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        pass
        
    # Attempt 3: Regex extraction of 'Answer' field
    match = re.search(r"['\"]Answer['\"]\s*:\s*(?:['\"]([^'\"]+)['\"]|\[['\"]?([^'\"\]]+)['\"]?\])", cleaned, re.IGNORECASE)
    if match:
        # Group 1 is simple string, Group 2 is list format like ['yes']
        val = match.group(1) or match.group(2)
        return {"Answer": val.strip()}

    return None

def iu_normalize_answer(raw_answer):
    """
    Normalizes the answer to 'yes', 'no', or 'abstain'.
    """
    if not isinstance(raw_answer, str):
        return None
    
    clean = raw_answer.lower().strip()
    
    # Check for abstain first
    if "abstain" in clean:
        return "abstain"
    
    # Check for yes/no
    if clean in ["yes", "no"]:
        return clean
        
    # Handle "yes." or "no."
    clean_nodot = clean.rstrip('.')
    if clean_nodot in ["yes", "no"]:
        return clean_nodot
        
    # Additional safety: check if starts with
    if clean.startswith("yes"):
        return "yes"
    if clean.startswith("no"):
        return "no"
        
    return "abstain" # Default fallback

def process_iu_xray(input_path, output_path):
    print(f"Processing {input_path} (Logic: IU X-Ray)...")
    
    valid_count = 0
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
                
            entry = json.loads(line)
            response = entry.get('response', '')
            
            parsed_json = iu_parse_json_response(response)
            
            final_ans = None
            extracted_reason = "json_parse_error"
            
            if parsed_json and 'Answer' in parsed_json:
                raw_ans = parsed_json['Answer']
                if isinstance(raw_ans, list) and len(raw_ans) > 0:
                    raw_ans = raw_ans[0]
                    
                final_ans = iu_normalize_answer(str(raw_ans))
                extracted_reason = "structured_parse"
            else:
                # Fallback: legacy keyword search (very strict)
                norm_resp = response.lower()
                # Simple direct check for "Yes." or "No." 
                clean_resp = response.lower().strip().rstrip('.')
                if clean_resp in ["yes", "no"]:
                    final_ans = clean_resp
                    extracted_reason = "fallback_direct_string"
                elif "answer: yes" in norm_resp or "'answer': 'yes'" in norm_resp or "'answer': ['yes']" in norm_resp:
                    final_ans = "yes"
                    extracted_reason = "fallback_keyword"
                elif "answer: no" in norm_resp or "'answer': 'no'" in norm_resp or "'answer': ['no']" in norm_resp:
                    final_ans = "no"
                    extracted_reason = "fallback_keyword"
                elif "abstain" in norm_resp:
                    final_ans = "abstain"
                    extracted_reason = "fallback_keyword_abstain"
                else:
                    # Final fallback: Treat as abstain if totally unparseable
                    final_ans = "abstain" 
                    extracted_reason = "parse_failed_default_abstain"

            # Update entry
            entry['extracted_answer'] = final_ans
            entry['extraction_method'] = extracted_reason
            
            # Logic for is_abstention
            entry['is_abstention'] = (final_ans == 'abstain')
            
            fout.write(json.dumps(entry) + "\n")
            valid_count += 1
            
    print(f"Finished. Processed {valid_count} lines to {output_path}")


# ==========================================
# LOGIC FOR Ham10000 
# ==========================================

def ham_parse_response(response_str):
    if not response_str:
        return 'abstain'
    
    # Normalize
    s = response_str.strip()
    
    # Strategy 1: Look for "Answer: X" or "Answer': X" or "Answer': [X]"
    match = re.search(r"Answer['\"]?\s*:\s*[\[\'\"\s]*([A-E])", s, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # Strategy 2: If simple string like "Option A" or just "A" at start
    # But be formatting aware.
    # Check for "Abstain" explicitly
    if 'abstain' in s.lower():
        return 'abstain'
        
    return 'abstain' # Default


def ham_parse_ground_truth(gt_str):
    # Format "C:trunk" -> "C"
    if not gt_str: return None
    return gt_str.split(':')[0].strip().upper()

def process_ham10000(input_path, output_path):
    print(f"Processing {input_path} -> {output_path} (Logic: HAM10000)")
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data.append(json.loads(line))
            
    processed_data = []
    for row in data:
        # Extract response
        resp = row.get('response', '')
        extracted = ham_parse_response(resp)
        
        # Handle "Abstain" mapping
        # In HAM10000, "E" is "Abstain to answer".
        
        normalized_answer = extracted
        if extracted == 'E':
            normalized_answer = 'abstain'
            
        row['extracted_answer'] = normalized_answer # Use 'abstain' keyword for compatibility
        row['raw_extracted_char'] = extracted
        
        # Parse GT
        gt = row.get('ground_truth', '')
        row['parsed_gt'] = ham_parse_ground_truth(gt)
        
        processed_data.append(row)
        
    with open(output_path, 'w') as f:
        for row in processed_data:
            f.write(json.dumps(row) + '\n')


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Answer Extraction Script")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--dataset", required=True, choices=['ham10000', 'iu_xray'], help="Dataset type to determine extraction logic")
    
    args = parser.parse_args()
    
    if args.dataset == 'iu_xray':
        process_iu_xray(args.input, args.output)
    elif args.dataset == 'ham10000':
        process_ham10000(args.input, args.output)
