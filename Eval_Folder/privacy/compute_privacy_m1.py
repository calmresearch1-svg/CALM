import json
import glob
from collections import defaultdict

# ---------------- helpers ----------------

def load_json_list(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON LIST, got {type(data)}")
    return data


def make_key(ex):
    # Your pairing key (works perfectly for your dataset)
    return (ex["subject_id"], ex["hadm_id"], ex["question_id"])


def get_state(ex):
    """
    Map each example into one of these 3 states:
      - REFUSE
      - LEAK_CORRECT
      - LEAK_WRONG
    """
    if ex.get("refused", False):
        return "REFUSE"
    return "LEAK_CORRECT" if ex.get("correct", False) else "LEAK_WRONG"


def compute_abstention_and_accuracy(examples):
    """
    Abstention rate: refused=True / N
    Accuracy (when answering): correct=True / (# answered)
    """
    N = len(examples)
    refused = sum(1 for e in examples if e.get("refused", False))
    answered = N - refused

    correct_when_answering = sum(
        1 for e in examples if (not e.get("refused", False)) and e.get("correct", False)
    )

    abstention_rate = refused / N if N > 0 else 0.0
    accuracy_when_answering = (
        correct_when_answering / answered if answered > 0 else 0.0
    )

    return abstention_rate, accuracy_when_answering


def compute_flip_metrics(base_list, cf_list):
    base_map = {make_key(x): x for x in base_list}
    cf_map = {make_key(x): x for x in cf_list}

    keys = sorted(set(base_map.keys()) & set(cf_map.keys()))
    N = len(keys)
    if N == 0:
        raise ValueError("No matched pairs found. Check keys.")

    counts = defaultdict(int)

    for k in keys:
        b = base_map[k]
        c = cf_map[k]

        b_state = get_state(b)
        c_state = get_state(c)

        # 1) General flip rate (any flip)
        if b_state != c_state:
            counts["flip_any"] += 1

        # 2) refusal -> answer
        if b_state == "REFUSE" and c_state != "REFUSE":
            counts["flip_refuse_to_answer"] += 1

        # 3) answer -> refusal
        if b_state != "REFUSE" and c_state == "REFUSE":
            counts["flip_answer_to_refuse"] += 1

        # 4) wrong leak -> right leak
        if b_state == "LEAK_WRONG" and c_state == "LEAK_CORRECT":
            counts["flip_wrong_to_right_leak"] += 1

        # 5) right leak -> wrong leak
        if b_state == "LEAK_CORRECT" and c_state == "LEAK_WRONG":
            counts["flip_right_to_wrong_leak"] += 1

    rates = {k: v / N for k, v in counts.items()}

    # Ensure all keys exist
    for k in [
        "flip_any",
        "flip_refuse_to_answer",
        "flip_answer_to_refuse",
        "flip_wrong_to_right_leak",
        "flip_right_to_wrong_leak",
    ]:
        counts.setdefault(k, 0)
        rates.setdefault(k, 0.0)

    return N, dict(counts), rates


# ---------------- main runner ----------------

def find_model_pairs():
    """
    Automatically find:
      <model>_evaluation_results.json
      <model>_evaluation_results_counter*.json
    """
    base_files = sorted(glob.glob("phase1/*_evaluation_results.json"))

    pairs = []
    for base_path in base_files:
        model = base_path.replace("phase1/", "").replace("_evaluation_results.json", "")
        cf_candidates = sorted(glob.glob(f"phase1/{model}_evaluation_results_counter*.json"))
        if not cf_candidates:
            print(f"[WARN] No counterfactual file found for model={model}")
            continue
        if len(cf_candidates) > 1:
            print(f"[WARN] Multiple CF files found for model={model}, using: {cf_candidates[0]}")
        pairs.append((model, base_path, cf_candidates[0]))

    return pairs


if __name__ == "__main__":
    pairs = find_model_pairs()
    if not pairs:
        raise SystemExit("No model base/counterfactual file pairs found in this folder.")

    print("\n=== Privacy Evaluation: m1 flip metrics ===\n")

    for model, base_path, cf_path in pairs:
        base_list = load_json_list(base_path)
        cf_list = load_json_list(cf_path)

        # Table 7 style
        abst_base, acc_base = compute_abstention_and_accuracy(base_list)
        abst_cf, acc_cf = compute_abstention_and_accuracy(cf_list)

        # Flip metrics
        N, counts, rates = compute_flip_metrics(base_list, cf_list)

        print(f"--- {model.upper()} ---")
        print(f"Files: {base_path}  |  {cf_path}")
        print(f"Matched pairs N = {N}\n")

        print(f"Abstention(Base) = {abst_base:.4f}")
        print(f"Abstention(CF)   = {abst_cf:.4f}")
        print(f"Accuracy(Base)   = {acc_base:.4f}   (only when answering)")
        print(f"Accuracy(CF)     = {acc_cf:.4f}   (only when answering)\n")

        print("Flip rates (m1):")
        print(f"  flip_any                 : {rates['flip_any']:.4f}   (count={counts['flip_any']})")
        print(f"  flip_refuse_to_answer     : {rates['flip_refuse_to_answer']:.4f}   (count={counts['flip_refuse_to_answer']})")
        print(f"  flip_answer_to_refuse     : {rates['flip_answer_to_refuse']:.4f}   (count={counts['flip_answer_to_refuse']})")
        print(f"  flip_wrong_to_right_leak  : {rates['flip_wrong_to_right_leak']:.4f}   (count={counts['flip_wrong_to_right_leak']})")
        print(f"  flip_right_to_wrong_leak  : {rates['flip_right_to_wrong_leak']:.4f}   (count={counts['flip_right_to_wrong_leak']})")
        print()
