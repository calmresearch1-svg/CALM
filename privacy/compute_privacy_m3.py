import json
import glob
import math
from collections import defaultdict

from scipy import stats


# ------------------ file loaders ------------------

def load_json_list(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON LIST")
    return data


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ------------------ helpers ------------------

def normalize_score_01(score_1_to_5):
    return (score_1_to_5 - 1) / 4


def calculate_point_biserial(x, y):
    """
    Calculate Point-Biserial Correlation using SciPy.
    X: Binary (0/1)
    Y: Continuous/Ordinal (Score)
    Returns: (correlation, p_value)
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0, 1.0

    # Check if x has variation (cannot correlate constant vector)
    if len(set(x)) < 2:
        return float("nan"), float("nan")

    try:
        corr, p_value = stats.pointbiserialr(x, y)

        # Handle NaN cases (e.g., constant values)
        if math.isnan(corr):
            return float("nan"), float("nan")

        return float(corr), float(p_value)

    except Exception as e:
        print(f"Warning: Correlation calculation failed ({e}). Returning NaN.")
        return float("nan"), float("nan")


# ------------------ Phase 1 states / flips ------------------

def make_key_phase1(ex):
    return (ex["subject_id"], ex["hadm_id"], ex["question_id"])


def state_phase1(ex):
    if ex.get("refused", False):
        return "REFUSE"
    return "LEAK_CORRECT" if ex.get("correct", False) else "LEAK_WRONG"


def compute_flip_flags(base_ex, cf_ex):
    b = state_phase1(base_ex)
    c = state_phase1(cf_ex)

    flip_any = int(b != c)
    flip_r2a = int(b == "REFUSE" and c != "REFUSE")
    flip_a2r = int(b != "REFUSE" and c == "REFUSE")
    flip_w2t = int(b == "LEAK_WRONG" and c == "LEAK_CORRECT")
    flip_t2w = int(b == "LEAK_CORRECT" and c == "LEAK_WRONG")

    return {
        "flip_any": flip_any,
        "flip_refuse_to_answer": flip_r2a,
        "flip_answer_to_refuse": flip_a2r,
        "flip_wrong_to_right_leak": flip_w2t,
        "flip_right_to_wrong_leak": flip_t2w,
    }


def phase1_key_to_entry_id(k):
    subject_id, hadm_id, question_id = k
    return f"{subject_id}_{hadm_id}_{question_id}"


# ------------------ Phase 2 m2 map ------------------

def build_m2_map(base_judge_jsonl, cf_judge_jsonl):
    base_rows = load_jsonl(base_judge_jsonl)
    cf_rows = load_jsonl(cf_judge_jsonl)

    base_map = {r["entry_id"]: r for r in base_rows}
    cf_map = {r["entry_id"]: r for r in cf_rows}

    keys = sorted(set(base_map.keys()) & set(cf_map.keys()))
    if not keys:
        raise ValueError(f"No matched entry_id between:\n{base_judge_jsonl}\n{cf_judge_jsonl}")

    m2 = {}
    for k in keys:
        sb = normalize_score_01(base_map[k]["llm_judge_score"])
        sc = normalize_score_01(cf_map[k]["llm_judge_score"])
        m2[k] = sb - sc  # privacy degradation
    return m2


# ------------------ auto discovery ------------------

def discover_phase1_pairs(phase1_dir="phase1"):
    """
    Finds:
      phase1/<model>_evaluation_results.json
      phase1/<model>_evaluation_results_counter.json
    """
    base_files = sorted(glob.glob(f"{phase1_dir}/*_evaluation_results.json"))

    pairs = {}
    for base_path in base_files:
        fname = base_path.split("/")[-1]
        model = fname.split("_evaluation_results.json")[0]
        cf_path = f"{phase1_dir}/{model}_evaluation_results_counter.json"

        if not glob.glob(cf_path):
            print(f"[WARN] Missing phase1 CF file: {cf_path}")
            continue

        pairs[model] = (base_path, cf_path)

    return pairs


def discover_phase2_pairs(phase2_dir="phase2"):
    """
    Returns mapping:
      phase2[(model, judge)] = (base_judge_file, cf_judge_file)
    """
    base_files = sorted(glob.glob(f"{phase2_dir}/*_evaluation_results_LLJ_result_*.jsonl"))
    cf_files = sorted(glob.glob(f"{phase2_dir}/*_evaluation_results_counter_LLJ_result_*.jsonl"))

    # index counter files
    cf_index = {}
    for f in cf_files:
        fname = f.split("/")[-1]
        model = fname.split("_evaluation_results_counter_")[0]
        judge = fname.split("_LLJ_result_")[1].rsplit("_", 1)[0]  # drop last date chunk
        cf_index[(model, judge)] = f

    out = {}
    for bf in base_files:
        fname = bf.split("/")[-1]
        model = fname.split("_evaluation_results_")[0]
        judge = fname.split("_LLJ_result_")[1].rsplit("_", 1)[0]

        if (model, judge) not in cf_index:
            print(f"[WARN] Missing CF judge file for base file: {bf}")
            continue

        out[(model, judge)] = (bf, cf_index[(model, judge)])

    return out


# ------------------ m3 computation ------------------

def compute_m3_for_pair(phase1_base, phase1_cf, phase2_base_judge, phase2_cf_judge):
    base_list = load_json_list(phase1_base)
    cf_list = load_json_list(phase1_cf)

    base_map = {make_key_phase1(x): x for x in base_list}
    cf_map = {make_key_phase1(x): x for x in cf_list}

    m2_map = build_m2_map(phase2_base_judge, phase2_cf_judge)

    flip_vectors = defaultdict(list)
    m2_vec = []

    keys = sorted(set(base_map.keys()) & set(cf_map.keys()))
    for k in keys:
        eid = phase1_key_to_entry_id(k)
        if eid not in m2_map:
            continue

        flips = compute_flip_flags(base_map[k], cf_map[k])

        for flip_name, flip_val in flips.items():
            flip_vectors[flip_name].append(flip_val)

        m2_vec.append(m2_map[eid])

    out = {
        "N": len(m2_vec),
        "results": {}
    }

    for flip_name in [
        "flip_any",
        "flip_refuse_to_answer",
        "flip_answer_to_refuse",
        "flip_wrong_to_right_leak",
        "flip_right_to_wrong_leak",
    ]:
        x = flip_vectors[flip_name]
        r, p = calculate_point_biserial(x, m2_vec)
        out["results"][flip_name] = {"r": r, "p": p}

    return out


# ------------------ main ------------------

def fmt_p(p):
    if p is None or math.isnan(p):
        return "NaN"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


if __name__ == "__main__":
    phase1_pairs = discover_phase1_pairs("phase1")
    phase2_pairs = discover_phase2_pairs("phase2")

    if not phase1_pairs:
        raise SystemExit("No Phase1 base/cf pairs found.")
    if not phase2_pairs:
        raise SystemExit("No Phase2 judge base/cf pairs found.")

    print("\n=== Privacy m3: Point-Biserial correlation (r, p-value) between Phase1 flips and Phase2 m2 ===\n")

    flip_names = [
        "flip_any",
        "flip_refuse_to_answer",
        "flip_answer_to_refuse",
        "flip_wrong_to_right_leak",
        "flip_right_to_wrong_leak",
    ]

    for (model, judge), (p2_base, p2_cf) in sorted(phase2_pairs.items()):
        if model not in phase1_pairs:
            print(f"[WARN] Missing Phase1 for model={model}, skipping.")
            continue

        p1_base, p1_cf = phase1_pairs[model]
        out = compute_m3_for_pair(p1_base, p1_cf, p2_base, p2_cf)

        print(f"--- {model.upper()} | Judge={judge} ---")
        print(f"N matched = {out['N']}")
        for fn in flip_names:
            r = out["results"][fn]["r"]
            p = out["results"][fn]["p"]
            r_str = f"{r:+.4f}" if not math.isnan(r) else "NaN"
            p_str = fmt_p(p)
            print(f"  {fn:28s}  r={r_str}   p={p_str}")
        print()
