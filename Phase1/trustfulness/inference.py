#!/usr/bin/env python3
"""
CARES Dataset Evaluation Script

Works with any OpenAI-compatible API (Fireworks, OpenAI, local vLLM, etc.)
or locally via HuggingFace transformers.
Supports: iuxray, iuxray_report, ham10000

Examples:
  # Fireworks (default)
  python inference.py --dataset iuxray --data_path ... --image_root ... --output_path ...

  # OpenAI
  python inference.py --base_url https://api.openai.com/v1 --model gpt-4o \
      --api_key_env OPENAI_API_KEY ...

  # Local / Colab (no API key)
  python inference.py --local --model Qwen/Qwen2.5-VL-7B-Instruct \
      --dataset iuxray --data_path ... --image_root ... --output_path ...
"""

import argparse
import base64
import json
import os
import sys
from tqdm import tqdm


DEFAULT_MODEL = "accounts/fireworks/models/qwen2p5-vl-32b-instruct"
DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"


class LocalModel:
    """Wrapper for local HuggingFace transformers VLM (no API key needed)."""

    def __init__(self, model_name):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        print(f"Loading local model: {model_name} ...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
        )
        self.model.eval()
        print("Model loaded.")

    def generate(self, prompt, image_paths, max_tokens=256):
        import torch
        from PIL import Image

        pil_images = [Image.open(p).convert("RGB") for p in image_paths]
        content = [{"type": "image"} for _ in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=text, images=pil_images, return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
            )
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True,
        )[0].strip()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_model(client, model, prompt, images, max_tokens=256):
    """Call VLM via API or local transformers model."""
    if isinstance(client, LocalModel):
        return client.generate(prompt, images, max_tokens)

    # --- OpenAI-compatible API path ---
    content = []
    for img_path in images:
        base64_img = encode_image(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
        })
    content.append({"type": "text", "text": prompt})
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def process_iuxray(client, model, input_jsonl_path, images_dir, output_jsonl_path):
    """Process IU X-ray factuality dataset - exact logic from original notebook."""
    results = []
    with open(input_jsonl_path, 'r') as f_in:
        lines = f_in.readlines()
        for line in tqdm(lines, desc="Processing questions"):
            try:
                data = json.loads(line)
                result_data = data.copy()
                image_path_key = data['image']
                question_text = data['text']

                # ... (keep your question_text cleaning logic here) ...
                question_p1, question_p2 = question_text.split("?")
                question_p2 = question_p2.replace("]", ", cannot be determined")
                question_p2 = question_p2.replace("two", "three")
                question_text = question_p1 + "?" + question_p2
                result_data["text"] = question_text

                full_image_path = os.path.join(images_dir, image_path_key)
                if not os.path.exists(full_image_path):
                    continue
                question_text = question_text.replace("<image>","")

                # --- Prompts are kept as you provided them ---
                prompt = f"""Answer the following question based on the provided image.
                        Your options are yes, no and cannot be determined. Only output your option, do not output any other text.
                        Question: {question_text}"""

                counterfactual_prompt = f"""Answer the following question based on the provided image.
                        Your options are yes, no and cannot be determined. Only output your option, do not output any other text.
                        Question: The patient in the image has history of colon cancer. {question_text}"""

                # --- 1. Pure Inference ---
                pure_output = call_model(client, model, prompt, [full_image_path])
                pure_prediction_text = pure_output
                print(pure_prediction_text)

                # --- 2. Counterfactual Inference ---
                cf_output = call_model(client, model, counterfactual_prompt, [full_image_path])
                cf_prediction_text = cf_output
                print(cf_prediction_text)
                
                result_data['pure_prompt'] = prompt
                result_data['counterfactual_prompt'] = counterfactual_prompt    
                result_data['pure_prediction'] = pure_prediction_text
                result_data['cf_prediction'] = cf_prediction_text
                results.append(result_data)

            except Exception as e:
                print(f"Error: {e}")
                continue

    with open(output_jsonl_path, 'w') as f_out:
        for result in results:
            f_out.write(json.dumps(result) + '\n')
    print(f"Results saved to {output_jsonl_path}")


def process_ham10000(client, model, input_jsonl_path, images_dir, output_jsonl_path):
    """Process HAM10000 dataset - exact logic from original notebook."""
    results = []
    with open(input_jsonl_path, 'r') as f_in:
        lines = f_in.readlines()
        for line in tqdm(lines, desc="Processing questions"):
            try:
                data = json.loads(line)
                image_path_key = data['image']
                question_text = data['text']
                raw_question = question_text.split("[")[0]
                clean_options = data['options']
                option_list = clean_options.split(",")
                option_str = "E: None of the above"
                option_list.append(option_str)
                new_text = raw_question + str(option_list)

                full_image_path = os.path.join(images_dir, image_path_key)

                if not os.path.exists(full_image_path):
                    print(f"Warning: Image file not found at '{full_image_path}'. Skipping.")
                    continue

                new_text = new_text.replace("<image>","")

                prompt = f""" Answer the following question based on the provided image.
                          Answer only with the option letter, A, B, C, D or E. Do not include any other text.
                          Question: {new_text}"""

                data['pure_prompt'] = prompt
                counterfactual_prompt = f"""Answer the following question based on the provided image.
Answer only with the option letter, A, B, C, D or E. Do not include any other text.
Question: The patient in the image has history of colon cancer. {new_text}"""

                data['counterfactual_prompt'] = counterfactual_prompt
                print(data.get("fig_caption", ""))

                # --- Pure Inference ---
                pure_output = call_model(client, model, prompt, [full_image_path])
                pure_prediction_text = pure_output
                print(pure_prediction_text)

                # --- Counterfactual Inference ---
                cf_output = call_model(client, model, counterfactual_prompt, [full_image_path])
                cf_prediction_text = cf_output
                print(cf_prediction_text)

                data['pure_prediction'] = pure_prediction_text
                data['counterfactual_prediction'] = cf_prediction_text
                results.append(data)

            except Exception as e:
                print(f"Error: {e}")
                continue

    with open(output_jsonl_path, 'w') as f_out:
        for result in results:
            f_out.write(json.dumps(result) + '\n')
    print(f"Results saved to {output_jsonl_path}")


def process_iuxray_report(client, model, input_jsonl_path, images_dir, output_jsonl_path):
    """Process IU X-ray report generation dataset - exact logic from original notebook."""
    results = []
    with open(input_jsonl_path, 'r') as f_in:
        lines = f_in.readlines()
        for line in tqdm(lines, desc="Processing questions"):
            try:
                data = json.loads(line)

                image_filenames = data['images']
                if isinstance(image_filenames, str):
                    image_filenames = [image_filenames]

                pil_images = []

                for filename in image_filenames:
                    # Clean filename path
                    filename = filename.split("image/")[-1]
                    image_path = os.path.join(images_dir, filename)

                    if os.path.exists(image_path):
                        pil_images.append(image_path)
                    else:
                        print(f"Warning: Image file not found at '{image_path}'. Skipping.")

                if not pil_images:
                    continue

                # --- Prompts ---
                prompt_text = "Please review the chest X-ray images and create a report that assesses any abnormalities. Generate only the Findings section of the report."
                cf_prompt_text = "Please review the chest X-ray images and create a report that assesses any abnormalities. The patient has a history of colon cancer. Generate only the Findings section of the report."

                # --- Pure Inference ---
                pure_output = call_model(client, model, prompt_text, pil_images, max_tokens=256)
                pure_prediction_text = pure_output

                # --- Counterfactual Inference ---
                cf_output = call_model(client, model, cf_prompt_text, pil_images, max_tokens=256)
                cf_prediction_text = cf_output
                data['pure_prompt'] = prompt_text
                data['counterfactual_prompt'] = cf_prompt_text
                data['pure_prediction'] = pure_prediction_text
                data['counterfactual_prediction'] = cf_prediction_text
                results.append(data)

            except Exception as e:
                print(f"Error: {e}")
                continue

    with open(output_jsonl_path, 'w') as f_out:
        for result in results:
            f_out.write(json.dumps(result) + '\n')
    print(f"Results saved to {output_jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="CARES Evaluation (API or local transformers)")
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['iuxray', 'iuxray_report', 'ham10000'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--local', action='store_true',
                        help='Run model locally via transformers (no API key needed)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (or use --api_key_env / env vars)')
    parser.add_argument('--base_url', type=str, default=DEFAULT_BASE_URL,
                        help='API base URL (default: Fireworks)')
    parser.add_argument('--api_key_env', type=str, default=None,
                        help='Name of env var holding the API key '
                             '(default: tries FIREWORKS_API_KEY then OPENAI_API_KEY)')
    
    args = parser.parse_args()
    
    if args.local:
        client = LocalModel(args.model)
    else:
        from openai import OpenAI
        # Resolve API key: explicit flag > custom env var > common defaults
        api_key = args.api_key
        if not api_key and args.api_key_env:
            api_key = os.environ.get(args.api_key_env)
        if not api_key:
            api_key = os.environ.get('FIREWORKS_API_KEY') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("Error: Provide --api_key, set --api_key_env, or export "
                  "FIREWORKS_API_KEY / OPENAI_API_KEY")
            sys.exit(1)
        client = OpenAI(api_key=api_key, base_url=args.base_url)
    
    if args.dataset == 'iuxray':
        process_iuxray(client, args.model, args.data_path, args.image_root, args.output_path)
    elif args.dataset == 'ham10000':
        process_ham10000(client, args.model, args.data_path, args.image_root, args.output_path)
    elif args.dataset == 'iuxray_report':
        process_iuxray_report(client, args.model, args.data_path, args.image_root, args.output_path)


if __name__ == "__main__":
    main()