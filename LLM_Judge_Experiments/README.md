# LLM Judge Framework

A modular and extensible Python framework for evaluating Large Language Model (LLM) responses using the LLM-as-a-Judge methodology. Built with LangChain for seamless integration with multiple LLM providers.

## Overview

This framework enables automated evaluation of LLM-generated responses across multiple quality aspects:

- **Robustness** - Consistency under input perturbations
- **Fairness** - Absence of demographic bias  
- **Privacy** - Protection of sensitive information
- **Safety** - Resistance to harmful outputs (jailbreak, overcautious, toxicity)
- **Trustfulness** - Accuracy and reliability of information

## Installation

```bash
# Install dependencies
pip install langchain langchain-core langchain-google-genai langchain-openai
pip install python-dotenv tqdm pydantic

# Set up API keys in .env file
echo "GEMINI_API_KEY=your-key-here" > .env
echo "OPENAI_API_KEY=your-openai-key" >> .env
```

## Project Structure

```
LLM_Judge_Experiments/
├── llm_judge/                 # Core framework package
│   ├── __init__.py           # Package exports
│   ├── base_judge.py         # LLMJudge and AsyncLLMJudge classes
│   ├── config.py             # Aspect definitions and configurations
│   ├── data_handlers.py      # Dataset-specific data parsers
│   ├── models.py             # LangChain model factory
│   ├── parsers.py            # Response parsing utilities
│   ├── prompts.py            # Prompt templates
│   └── utils.py              # Helper functions
├── llj_robustness.py         # Robustness evaluation script
├── llj_fairness.py           # Fairness evaluation script
├── llj_privacy.py            # Privacy evaluation script
├── llj_safety.py             # Safety evaluation script
├── JSONs/                    # Input data directory
└── LLJ_Outputs/              # Output results directory
```

---

## Core Package: `llm_judge/`

### `base_judge.py` - Judge Classes

Contains the main evaluation classes:

#### `EvalTask` (dataclass)
Task structure for **combined evaluations** (robustness, fairness):
```python
@dataclass
class EvalTask:
    entry_id: str              # Unique identifier
    image_path: str            # Absolute path to image
    image_path_relative: str   # Relative path for output
    original_q: str            # Original question
    original_a: str            # Model's original response
    cf_q: str                  # Counterfactual question
    cf_a: str                  # Model's counterfactual response
    model_name: str            # Name of evaluated model
```

#### `SingleEvalTask` (dataclass)
Task structure for **single evaluations** (privacy, safety):
```python
@dataclass
class SingleEvalTask:
    entry_id: str              # Unique identifier
    image_path: str            # Optional image path
    image_path_relative: str   # Optional relative path
    question: str              # The question
    response: str              # Model's response
    model_name: str            # Name of evaluated model
```

#### `LLMJudge` (sync class)
Synchronous evaluator for simple use cases:
```python
from llm_judge import LLMJudge, ASPECT_DEFINITIONS

judge = LLMJudge(
    aspect_config=ASPECT_DEFINITIONS["robustness"],
    judge_model="gemini-2.5-flash",
    evidence_modality="chest X-ray image"
)

# Single evaluation
result = judge.evaluate_single(question, response, image_path)

# Combined evaluation (original vs counterfactual)
result = judge.evaluate_combined(orig_q, orig_a, cf_q, cf_a, image_path)
```

#### `AsyncLLMJudge` (async class)
High-performance concurrent evaluator with:
- Semaphore-based rate limiting
- Incremental file writing (fault-tolerant)
- Exponential backoff retry

```python
from llm_judge import AsyncLLMJudge, ASPECT_DEFINITIONS

judge = AsyncLLMJudge(
    aspect_config=ASPECT_DEFINITIONS["robustness"],
    judge_model="gemini-2.5-flash",
    evidence_modality="chest X-ray image",
    concurrency=10,    # Max concurrent requests
    max_retries=8      # Retry on failures
)

# Batch evaluation with progress bar
await judge.evaluate_batch(tasks, output_path)           # Combined tasks
await judge.evaluate_single_batch(tasks, output_path)    # Single tasks
```

---

### `config.py` - Configuration

Central configuration for aspects, datasets, and outputs.

#### Aspect Definitions
```python
from llm_judge import ASPECT_DEFINITIONS

# Access predefined aspects
robustness_config = ASPECT_DEFINITIONS["robustness"]
fairness_config = ASPECT_DEFINITIONS["fairness"]
privacy_config = ASPECT_DEFINITIONS["privacy"]
safety_jailbreak = ASPECT_DEFINITIONS["safety_jailbreak"]
safety_overcautious = ASPECT_DEFINITIONS["safety_overcautious"]
safety_toxicity = ASPECT_DEFINITIONS["safety_toxicity"]

# Each AspectConfig contains:
# - name: Aspect identifier
# - definition: Text definition for prompts
# - requires_image: Whether images are needed
# - evaluation_mode: "single" or "combined"
```

#### Dataset Configuration
```python
DATASET_IMAGE_ROOTS = {
    "iu_xray": Path("datasets/iu_xray"),
    "ham10000": Path("datasets"),
    "harvard_fairvlmed": Path("datasets/harvard_fairvlmed/Test"),
}

DATASET_EVIDENCE_MODALITY = {
    "iu_xray": "chest X-ray image",
    "ham10000": "dermatoscopy image",
    "harvard_fairvlmed": "fundus image",
}
```

---

### `data_handlers.py` - Dataset Handlers

Abstract base class with dataset-specific implementations for parsing different JSONL formats.

```python
from llm_judge import get_data_handler

# Get handler for specific dataset
handler = get_data_handler("iu_xray")

# Available handlers:
# - IUXrayHandler
# - HAM10000Handler
# - HarvardFairVLMedHandler

# Common interface methods:
entry_id = handler.get_entry_id(entry)
question = handler.get_question(entry)
response = handler.get_response(entry)
image_path = handler.get_full_image_path(entry)
```

---

### `models.py` - Model Factory

Unified interface for creating LangChain LLM instances:

```python
from llm_judge.models import ModelFactory

# Automatically detects provider from model name
model = ModelFactory.create("gemini-2.5-flash")
model = ModelFactory.create("gpt-4o-mini")
model = ModelFactory.create("gpt-4o", temperature=0.1)

# Supported models:
# Gemini: gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
# OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo
```

---

### `parsers.py` - Response Parsing

Utilities for extracting structured data from LLM responses:

```python
from llm_judge.parsers import extract_score, parse_judge_response

# Extract just the score (1-5)
score = extract_score(raw_response)  # Returns int or None

# Parse full response with reasoning
result = parse_judge_response(raw_response)
# Returns: {"reasoning": "...", "score": 4}
```

Features:
- JSON parsing with fallback regex
- Automatic markdown code block removal
- Robust error handling

---

### `prompts.py` - Prompt Templates

Two main prompt templates using LangChain's `PromptTemplate`:

```python
from llm_judge.prompts import PromptBuilder

builder = PromptBuilder(
    aspect_instruction="You are a robustness evaluator...",
    output_format='{"reasoning": "...", "score": X}'
)

# For single evaluations (privacy, safety)
prompt = builder.build_single_prompt(question, response)

# For combined evaluations (robustness, fairness)
prompt = builder.build_combined_prompt(orig_q, orig_a, cf_q, cf_a)
```

---

### `utils.py` - Utility Functions

Helper functions for common operations:

```python
from llm_judge.utils import (
    calculate_statistics,    # Calculate mean/std from output file
    print_statistics,        # Pretty-print statistics
    build_multimodal_content, # Create LangChain multimodal messages
    load_jsonl,              # Load JSONL files
    save_jsonl_entry,        # Append to JSONL files
    encode_image_base64,     # Encode images for API calls
)

# Calculate and print evaluation statistics
stats = calculate_statistics("output.jsonl")
print(f"Mean: {stats['mean']}, Std: {stats['std']}")

# Or use the formatted printer
print_statistics("output.jsonl")
```

---

## Usage Examples

### Robustness Evaluation

Compares model responses to original and counterfactual inputs:

```bash
python llj_robustness.py \
    --original_file JSONs/Robustness/qwen_original_results.jsonl \
    --cf_file JSONs/Robustness/qwen_cf_wrong_region_results.jsonl \
    --dataset iu_xray \
    --judge_model gemini-2.5-flash \
    --concurrency 10
```

### Fairness Evaluation

Evaluates consistency across demographic variations:

```bash
python llj_fairness.py \
    --input_file JSONs/Fairness/FairVLMed/qwen_merged_results.jsonl \
    --dataset harvard_fairvlmed \
    --judge_model gemini-2.5-flash \
    --concurrency 5 \
    --resume  # Resume from interrupted evaluation
```

### Privacy Evaluation

Assesses privacy protection in responses:

```bash
python llj_privacy.py \
    --input_file JSONs/Privacy/Base/qwen_evaluation_results.json \
    --judge_model gemini-2.5-flash \
    --concurrency 5
```

### Safety Evaluation

Evaluates safety aspects (jailbreak, overcautious, toxicity):

```bash
# Jailbreak resistance evaluation
python llj_safety.py \
    --input_file JSONs/Safety/jailbreak/qwen_1_cnt.json \
    --aspect jailbreak \
    --judge_model gemini-2.5-flash

# Toxicity evaluation  
python llj_safety.py \
    --input_file JSONs/Safety/toxicity/qwen_original.json \
    --aspect toxicity \
    --judge_model gemini-2.5-flash

# Overcautious evaluation
python llj_safety.py \
    --input_file JSONs/Safety/overcautious/qwen_results.json \
    --aspect overcautious \
    --dry_run  # Preview without running
```

---

## Output Format

Results are saved as JSONL files with the following structure:

```json
{
    "entry_id": "123",
    "image_path": "CXR123_IM-456/0.png",
    "original_question": "Does the X-ray show...",
    "original_response": "Based on the image...",
    "counterfactual_question": "Does the X-ray show...",
    "counterfactual_response": "The image indicates...",
    "llm_judge_response": "{\"reasoning\": \"...\", \"score\": 4}",
    "llm_judge_score": 4
}
```

### Statistics

After evaluation, view statistics with:

```bash
python llj_robustness.py --stats_only LLJ_Outputs/Robustness/results.jsonl
```

Output:
```
==================================================
LLM JUDGE EVALUATION STATISTICS
==================================================
File: LLJ_Outputs/Robustness/results.jsonl
Total entries: 500
Valid scores: 498
Mean score: 3.8542
Std deviation: 0.9123
==================================================
```

---

## Common CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input_file`, `-i` | Path to input data file | Required |
| `--judge_model`, `-m` | LLM to use as judge | `gemini-2.5-flash` |
| `--output_file`, `-o` | Custom output path | Auto-generated |
| `--concurrency` | Concurrent API calls | `5-10` |
| `--start_index`, `-s` | Start processing from index | `0` |
| `--end_index`, `-e` | Stop processing at index | End of file |
| `--limit`, `-l` | Max entries to process | All |
| `--dry_run` | Preview without API calls | `False` |
| `--resume`, `-r` | Skip already completed entries | `False` |

---

## Extending the Framework

### Adding a New Aspect

1. Define the aspect in `config.py`:
```python
ASPECT_DEFINITIONS["my_aspect"] = AspectConfig(
    name="my_aspect",
    definition="Your aspect definition here...",
    requires_image=True,
    evaluation_mode="combined"  # or "single"
)
```

2. Create a new evaluation script (use existing scripts as templates).

### Adding a New Dataset

1. Add image root and evidence modality to `config.py`:
```python
DATASET_IMAGE_ROOTS["my_dataset"] = Path("datasets/my_dataset")
DATASET_EVIDENCE_MODALITY["my_dataset"] = "CT scan image"
```

2. Create a handler in `data_handlers.py`:
```python
class MyDatasetHandler(BaseDataHandler):
    def __init__(self, image_root=None):
        super().__init__("my_dataset", image_root)
    
    def get_entry_id(self, entry): ...
    def get_question(self, entry): ...
    def get_response(self, entry): ...
    def get_image_path(self, entry): ...
```

3. Register in the factory function:
```python
handlers["my_dataset"] = MyDatasetHandler
```

---
