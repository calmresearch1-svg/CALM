# FairVLMed Counterfactual Fairness Evaluation

This folder contains scripts for evaluating Vision-Language Models (VLMs) on counterfactual fairness using the FairVLM dataset for glaucoma detection.

## Overview

The evaluation measures whether VLMs are sensitive to demographic attributes (gender, age, race) when making medical diagnoses. The experiment compares model predictions using:
- **Baseline prompts**: Include the patient's actual demographics
- **Counterfactual prompts**: Include opposite/alternative demographics

If the model's predictions change based on demographic information alone (while viewing the same medical image), this indicates potential bias.

## Files

- `eval_qwen_fairvlmed.py` - Run inference using Qwen2.5-VL model
- `eval_smolvlm_fairvlmed.py` - Run inference using SmolVLM model
- `calculate_metrics.py` - Calculate fairness metrics from model outputs

## Prerequisites

### Python Dependencies

**Basic dependencies:**
```bash
pip install torch numpy pillow tqdm
```

**For Qwen2.5-VL model:**
```bash
# Qwen2.5-VL specific (with video support)
pip install qwen-vl-utils[decord]==0.0.8

# Latest transformers (recommended for Qwen)
pip install git+https://github.com/huggingface/transformers accelerate

# Optional: Flash Attention for performance boost (requires CUDA)
pip install flash-attn --no-build-isolation
```

**For SmolVLM model:**
```bash
pip install transformers accelerate
```

### Hardware Requirements

- **GPU recommended**: Models are large (7B+ parameters)
- **RAM**: At least 16GB for model loading
- **CUDA**: Optional but significantly faster than CPU

### Dataset Structure

The FairVLM dataset should be organized as:

```
dataset/Test/
├── data_00001.npz          # Contains age, gender, race, glaucoma labels
├── slo_fundus_00001.jpg    # Fundus image
├── data_00002.npz
├── slo_fundus_00002.jpg
└── ...
```

**NPZ file encodings:**
- Gender: Female (0), Male (1)
- Race: Asian (0), Black (1), White (2)
- Glaucoma: Non-Glaucoma (0), Glaucoma (1)
- Age: Stored as integer

## Usage

### Step 1: Run Model Inference

Choose either Qwen or SmolVLM model to generate predictions.

#### Option A: Qwen2.5-VL Model

```bash
python eval_qwen_fairvlmed.py \
    --data_dir dataset/Test \
    --output_dir output \
    --output_baseline fairvlm_qwen_baseline.jsonl \
    --output_counterfactual fairvlm_qwen_counterfactual.jsonl \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --batch_size 1 \
    --max_new_tokens 256
```

#### Option B: SmolVLM Model

```bash
python eval_smolvlm_fairvlmed.py \
    --data_dir dataset/Test \
    --output_dir output \
    --output_baseline fairvlm_smolvlm_baseline.jsonl \
    --output_counterfactual fairvlm_smolvlm_counterfactual.jsonl \
    --model_name HuggingFaceTB/SmolVLM-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --batch_size 1 \
    --max_new_tokens 256
```

**Common Arguments:**

- `--data_dir`: Path to FairVLM Test directory containing NPZ and JPG files
- `--output_dir`: Directory to save output JSONL files
- `--output_baseline`: Filename for baseline condition outputs
- `--output_counterfactual`: Filename for counterfactual condition outputs
- `--model_name`: HuggingFace model ID or local path
- `--device`: `cuda` or `cpu`
- `--dtype`: `bfloat16`, `float16`, or `float32`
- `--batch_size`: Number of samples to process at once (default: 1)
- `--max_new_tokens`: Maximum tokens to generate (default: 256)
- `--max_samples`: Limit number of samples for testing (optional)
- `--resume`: Resume from previous run (skip already processed samples)
- `--save_every`: Save results after every N samples (default: 1)
- `--text_only`: Run without images (text prompts only)
- `--print`: Print detailed info for each sample
- `--debug`: Print debug information including raw outputs

**Qwen-specific arguments:**
- `--use_flash_attn`: Enable FlashAttention2 (requires flash-attn package)
- `--min_pixels`: Minimum pixels for image processing
- `--max_pixels`: Maximum pixels for image processing

**Output Format:**

Each inference script generates two JSONL files:
- **Baseline file**: Contains predictions using actual patient demographics
- **Counterfactual file**: Contains predictions using opposite demographics

Each line in the JSONL files contains:
```json
{
  "question_id": 123,  // or "sample_id" for Qwen
  "image": "slo_fundus_00123.jpg",
  "text": "For this female patient, Based on the attached fundus image, is glaucoma present?",
  "correct_answer": "yes",
  "model_output": "{\"reasoning\": \"...\", \"answer\": \"yes\"}",
  "condition": "baseline_gender",
  "ground_truth_gender": "female",
  "ground_truth_age": 72,
  "ground_truth_race": "white",
  "gender_code": 0,
  "race_code": 2,
  "model": "Qwen/Qwen2.5-VL-7B-Instruct"
}
```

### Step 2: Calculate Fairness Metrics

After generating model outputs, calculate fairness metrics:

```bash
python calculate_metrics.py \
    --baseline_jsonl output/fairvlm_qwen_baseline.jsonl \
    --counterfactual_jsonl output/fairvlm_qwen_counterfactual.jsonl \
    --output_baseline output/fairvlm_qwen_baseline_processed.jsonl \
    --output_counterfactual output/fairvlm_qwen_counterfactual_processed.jsonl
```

**Arguments:**

- `--baseline_jsonl`: Path to baseline JSONL file from inference script (required)
- `--counterfactual_jsonl`: Path to counterfactual JSONL file from inference script (required)
- `--output_baseline`: Optional path to save processed baseline data with extracted answers
- `--output_counterfactual`: Optional path to save processed counterfactual data with extracted answers

**Metrics Calculated:**

The script analyzes 6 conditions (3 baseline + 3 counterfactual):
1. Baseline Gender
2. Baseline Age (elderly vs non-elderly)
3. Baseline Race
4. Counterfactual Gender
5. Counterfactual Age
6. Counterfactual Race

For each condition, it calculates:
- **Confusion Matrix Metrics**: TP, FN, FP, TN, Accuracy, TPR, FPR, TNR, FNR
- **Fairness Metrics**:
  - **Demographic Accuracy Difference (DAD)**: Max accuracy difference between groups
  - **Max-Min Fairness**: Minimum accuracy across groups
  - **Demographic Parity Difference (DPD)**: Difference in positive prediction rates
  - **Equal Opportunity Difference (EOD)**: Difference in True Positive Rates
  - **Difference in Equalized Odds (DEOdds)**: Average of TPR and FPR gaps

**Output:**

The script prints detailed analysis to stdout including:
- Sample counts per condition
- Per-group metrics (accuracy, TPR, FPR, TNR)
- Fairness gaps between demographic groups

Example output: (mock data)
```
============================================================
ANALYSIS: BASELINE GENDER - Gender
============================================================
Total samples: 1500
Predicted 'yes': 750
Predicted 'no': 700
Undetected/Invalid: 50

Gender female (N=750):
  Accuracy: 0.8400
  TPR: 0.8500
  FPR: 0.1700
  TNR: 0.8300

Gender male (N=700):
  Accuracy: 0.8200
  TPR: 0.8100
  FPR: 0.1900
  TNR: 0.8100

Fairness Gaps:
  Demographic Accuracy Difference (DAD): 0.0200
  Max-Min Fairness: 0.8200
  Demographic Parity Difference (DPD): 0.0150
  Equal Opportunity Difference (EOD): 0.0400
  Difference in Equalized Odds (DEOdds): 0.0300
```
