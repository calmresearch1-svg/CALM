import os
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq, Qwen2_5_VLForConditionalGeneration
from skimage.util import random_noise
import numpy as np
import warnings
from tqdm.auto import tqdm

# Suppress a specific PIL warning about decompression bombs
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# --- 1. VLM Handler Classes (Modular Model Loading) ---
# Base class to define the standard interface for any VLM
class BaseVLM:
    """Abstract base class for Vision Language Models."""
    def __init__(self, model_id, device='cuda'):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._load_model()
        print(f"Model {self.model_id} loaded successfully on {self.device}.")

    def _load_model(self):
        """Loads the model and tokenizer. To be implemented by subclasses."""
        raise NotImplementedError

    def get_response(self, image: Image.Image, prompt: str) -> str:
        """Generates a response for a given image and prompt."""
        raise NotImplementedError

# Implementation for Qwen-VL-Chat
class QwenVLM(BaseVLM):
    """Handler for Qwen-VL series models."""
    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16
        ).eval()

    def get_response(self, image: Image.Image, prompt: str, prompt_style: str = "default") -> str:

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text], 
            images=[image.convert('RGB')], 
            return_tensors="pt"
        ).to(self.device)

        try:
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            input_len = inputs['input_ids'].shape[1]
            response_ids = generated_ids[:, input_len:]
            response = self.processor.batch_decode(response_ids, skip_special_tokens=True)[0]
            return response.strip()
        except Exception as e:
            print(f"Error during Qwen inference: {e}")
            return "Error during inference."

# CORRECTED Implementation for SmolVLM (Idefics3-based)
class SmolVLM(BaseVLM):
    """Handler for SmolVLM, which is based on the Idefics3 architecture."""
    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=self.device
        )

    def get_response(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        
        templated_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=templated_prompt, images=[image], return_tensors="pt").to(self.device)

        try:
            generated_ids = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)
            generated_texts = self.processor.batch_decode(
                generated_ids[:, inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            return generated_texts[0].strip()
        except Exception as e:
            print(f"Error during SmolVLM inference: {e}")
            return "Error during inference."

# Implementation for Microsoft Phi-4-Vision
class Phi4VLM(BaseVLM):
    """Handler for Microsoft's Phi-4 Vision model."""
    def _load_model(self):
        # **FIX**: Changed loading strategy to avoid device_map issues with PEFT.
        # Load the model on CPU first, then move to the target device.
        # Also fixed the deprecation warning by using 'dtype' instead of 'torch_dtype'.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16, # Use bfloat16 for better performance
            _attn_implementation="eager"
        ).to(self.device) # Move to device *after* loading
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )

    def get_response(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"},
        ]
        
        prompt_string = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            prompt_string, 
            images=[image.convert('RGB')], 
            return_tensors="pt"
        ).to(self.device)

        try:
            generate_ids = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False # For reproducibility
            )
            # Remove input tokens from the generated output
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return response.strip()
        except Exception as e:
            print(f"Error during Phi-4 Vision inference: {e}")
            return "Error during inference."


def get_model(model_name: str, device: str = 'cuda'):
    """Factory function to get the appropriate VLM handler."""
    model_map = {
        "qwen": ("Qwen/Qwen2.5-VL-7B-Instruct", QwenVLM),
        "smolvlm": ("HuggingFaceTB/SmolVLM-Instruct", SmolVLM),
        "phi4": ("microsoft/Phi-4-multimodal-instruct", Phi4VLM),
    }
    if model_name.lower() in model_map:
        model_id, model_class = model_map[model_name.lower()]
        return model_class(model_id, device=device)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(model_map.keys())}")

# --- 2. Data and Noise Utilities ---

def load_dataset(dataset_path: str):
    """Loads the dataset from a .jsonl file."""
    with open(dataset_path, 'r') as f:
        for line in f:
            yield json.loads(line)

def apply_noise(image: Image.Image, noise_type: str) -> Image.Image:
    """Applies a specified type of noise to a PIL image."""
    if noise_type == 'original':
        return image
    
    img_array = np.array(image)
    
    if noise_type.startswith('gaussian'):
        # Following CARES paper's methodology: std is on the 0-255 pixel scale.
        std = float(noise_type.split('_')[-1])
        noise = np.random.normal(0, std, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    elif noise_type.startswith('salt_pepper'):
        # Using skimage for salt & pepper as it's convenient
        amount = float(noise_type.split('_')[-1])
        # random_noise converts integer images to float [0,1], adds noise,
        # so we need to scale back to [0,255].
        noisy_array_float = random_noise(img_array, mode='s&p', amount=amount)
        noisy_array = (noisy_array_float * 255).astype(np.uint8)
    else:
        print(f"Warning: Unknown noise type '{noise_type}'. Returning original image.")
        return image
        
    return Image.fromarray(noisy_array)


def add_abstention_option(question_text: str) -> str:
    """Adds abstention option to the question text by appending 'can not be inferred from the image' after 'no'."""
    # Find the word "no" and add the abstention phrase after it
    modified_text = question_text.replace("no", "no, can not be inferred from the image")
    return modified_text

# --- 3. Main Inference Function ---

def run_inference(config: dict):
    """
    Main function to run the inference pipeline.
    
    Args:
        config (dict): A dictionary containing configuration parameters:
            - model_name (str): The model to use ('qwen', 'smolvlm', 'phi4').
            - dataset_path (str): Full path to the .jsonl dataset file.
            - image_base_path (str): Base directory for images.
            - output_path (str): Path to save the output .jsonl file.
            - noise_types (list): List of noise types to apply (e.g., ['original', 'gaussian_6']).
            - device (str): 'cuda' or 'cpu'.
            - prompt_style (str): 'default' or 'add_abstention_option'.
    """
    print("--- Starting Inference Pipeline ---")
    print(f"Configuration: {config}")

    # Initialize model
    vlm = get_model(config['model_name'], device=config.get('device', 'cuda'))
    
    # Load dataset
    dataset = list(load_dataset(config['dataset_path']))
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    
    print(f"Running inference on {len(dataset)} data points...")
    
    # Open the output file in append mode to save progress
    with open(config['output_path'], 'a') as outfile:
        for item in tqdm(dataset, desc="Processing data"):
            original_image_path = os.path.join(config['image_base_path'], item['image'])
            
            if not os.path.exists(original_image_path):
                print(f"Warning: Image not found at {original_image_path}. Skipping.")
                continue

            try:
                original_image = Image.open(original_image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {original_image_path}: {e}. Skipping.")
                continue
            
            # Remove the <image> placeholder token from the question text
            question_text = item['question'].replace('<image>', '').strip()

            for noise_type in config['noise_types']:
                # Apply noise
                noisy_image = apply_noise(original_image, noise_type)
                
                # Add abstention option to the question text, if specified
                if config['prompt_style'] == "add_abstention_option":
                    question_text = add_abstention_option(question_text)
                
                # Get model response
                model_response = vlm.get_response(noisy_image, question_text)
                
                # Prepare result log
                result = {
                    "question_id": item.get("question_id"),
                    "question": item["question"], # Log the original question
                    "cleaned_question": question_text, # Log the cleaned question
                    "ground_truth_answer": item["answer"],
                    "image_path": item["image"],
                    "model_name": config['model_name'],
                    "noise_type": noise_type,
                    "model_response": model_response,
                }
                
                # Write result to file immediately
                outfile.write(json.dumps(result) + '\n')
                outfile.flush() # Ensure it's written to disk

    print(f"--- Inference Complete ---")
    print(f"Results saved to {config['output_path']}")

