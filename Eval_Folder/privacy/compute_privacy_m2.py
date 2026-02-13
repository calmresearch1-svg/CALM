import json
import glob
import math
from collections import defaultdict

# ------------------ helpers ------------------

def load_jsonl(path):
    """Load JSONL file into a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_score_01(score_1_to_5):
    """Map integer score in {1..5} to [0,1]."""
    return (score_1_to_5 - 1) / 4


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    """Sample standard deviation."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def parse_model_name_from_filename(path):
    """
    Example:
      llama_evaluation_results_LJJ_result_gpt-4-1-...jsonl
    -> model = llama
    """
    base = path.split("/")[-1]
    model = base.split("_evaluation_results")[0]
    return model


def parse_judge_name_from_filename(path):
    """
    Extract judge model name between 'LLJ_result_' and the date
    Example:
      ...LLJ_result_gpt-4-1-2025-04-14_2026-01-17.jsonl
    -> gpt-4-1-2025-04-14
    """
    base = path.split("/")[-1]
    marker = "_LLJ_result_"
    if marker not in base:
        return "unknown"
    after = base.split(marker, 1)[1]
    # remove trailing _YYYY-MM-DD.jsonl
    parts = after.split("_")
    # judge name can itself contain '-' so just remove last date chunk
    # files seem to end with _2026-01-17.jsonl
    judge = "_".join(parts[:-1]).replace(".jsonl", "")
    return judge


# ------------------ core m2 computation ------------------

def compute_m2_from_pair(base_file, cf_file):
    base_rows = load_jsonl(base_file)
    cf_rows = load_jsonl(cf_file)

    base_map = {r["entry_id"]: r for r in base_rows}
    cf_map = {r["entry_id"]: r for r in cf_rows}

    keys = sorted(set(base_map.keys()) & set(cf_map.keys()))
    if not keys:
        raise ValueError(f"No matched entry_id pairs between:\n{base_file}\n{cf_file}")

    m2_list = []

    for k in keys:
        sb = base_map[k]["llm_judge_score"]
        sc = cf_map[k]["llm_judge_score"]

        # normalize
        sbn = normalize_score_01(sb)
        scn = normalize_score_01(sc)

        # privacy degradation: base - cf
        m2_i = sbn - scn
        m2_list.append(m2_i)

    return {
        "N": len(keys),
        "m2_mean": mean(m2_list),
        "m2_std": stdev(m2_list),
        "m2_values": m2_list,  # optional if you want to reuse later
    }


def find_phase2_pairs(phase2_dir="phase2"):
    """
    Finds pairs like:
      <model>_evaluation_results_LLJ_result_<judge>_<date>.jsonl
      <model>_evaluation_results_counter_LLJ_result_<judge>_<date>.jsonl
    """
    base_files = sorted(glob.glob(f"{phase2_dir}/*_evaluation_results_LLJ_result_*.jsonl"))
    cf_files = sorted(glob.glob(f"{phase2_dir}/*_evaluation_results_counter_LLJ_result_*.jsonl"))

    # index counterfactual files by (model, judge)
    cf_index = {}
    for f in cf_files:
        model = parse_model_name_from_filename(f)
        judge = parse_judge_name_from_filename(f)
        cf_index[(model, judge)] = f

    pairs = []
    for bf in base_files:
        model = parse_model_name_from_filename(bf)
        judge = parse_judge_name_from_filename(bf)

        key = (model, judge)
        if key not in cf_index:
            print(f"[WARN] Missing CF file for base: model={model}, judge={judge}")
            continue

        pairs.append((model, judge, bf, cf_index[key]))

    return pairs


if __name__ == "__main__":
    pairs = find_phase2_pairs("phase2")
    if not pairs:
        raise SystemExit("No Phase2 base/counterfactual judge file pairs found inside ./phase2")

    print("\n=== Privacy Phase2: m2 (LLJ paired delta) ===\n")

    # store results
    results_by_model = defaultdict(list)

    for model, judge, base_file, cf_file in pairs:
        out = compute_m2_from_pair(base_file, cf_file)

        print(f"--- {model.upper()} | Judge={judge} ---")
        print(f"N = {out['N']}")
        print(f"m2_mean (base-cf, normalized) = {out['m2_mean']:.4f}")
        print(f"m2_std  (paired, normalized) = {out['m2_std']:.4f}")
        print()

        results_by_model[model].append(out["m2_mean"])

    # average across judges per evaluated model
    print("=== Average m2 across judges (per evaluated model) ===\n")
    for model, m2_means in results_by_model.items():
        print(f"{model.upper():10s}  mean_m2_across_judges = {mean(m2_means):.4f}")
