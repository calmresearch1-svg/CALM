"""
vlm_models.py

Shared module for VLM model definitions and factory function.
Contains BaseVLM, QwenVLM, SmolVLM, and FireworksQwenVLM classes.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import warnings
import base64
import io
import requests

warnings.filterwarnings("ignore")

class BaseVLM:
    """Abstract base class for Vision Language Models."""
    def __init__(self, model_id, device='cuda'):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()
        print(f"✓ Model {self.model_id} loaded successfully on {self.device}")

    def _load_model(self):
        """Loads the model and processor. To be implemented by subclasses."""
        raise NotImplementedError

    def get_response(self, image: Image.Image, prompt: str) -> str:
        """Generates a response for a given image and prompt."""
        raise NotImplementedError


class QwenVLM(BaseVLM):
    """Handler for Qwen2.5-VL series models."""
    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16
        ).eval()

    def get_response(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text], 
            images=[image.convert('RGB')], 
            return_tensors="pt"
        ).to(self.device)

        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1
                )
            
            input_len = inputs['input_ids'].shape[1]
            response_ids = generated_ids[:, input_len:]
            response = self.processor.batch_decode(
                response_ids, 
                skip_special_tokens=True
            )[0]
            return response.strip()
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return "Error during inference."


class SmolVLM(BaseVLM):
    """Handler for SmolVLM (Idefics3-based)."""
    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'
            
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16
        ).eval()

    def get_response(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        prompt_text = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=prompt_text, 
            images=[image.convert('RGB')], 
            return_tensors="pt"
        ).to(self.device)

        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
            
            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Extract only the assistant's response
            response = generated_texts[0]
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return "Error during inference."


class FireworksQwenVLM(BaseVLM):
    """Handler for Qwen2.5-VL 32B model via Fireworks AI API."""
    def _load_model(self):
        self.api_key = "YOUR_API_KEY"
        if not self.api_key:
            print("Warning: FIREWORKS_API_KEY environment variable not set. Inference will fail.")
        
        # We can use the model_id passed directly or map it
        if "fireworks" not in self.model_id and "32b" in self.model_id.lower():
             self.model_id = "accounts/fireworks/models/qwen2p5-vl-32b-instruct"
        
        self.api_url = "https://api.fireworks.ai/inference/v1/chat/completions"

    def get_response(self, image: Image.Image, prompt: str) -> str:
        if not self.api_key:
             return "Error: FIREWORKS_API_KEY not set."

        # Convert image to base64
        try:
            buffered = io.BytesIO()
            # Convert to RGB to ensure JPEG compatibility
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            print(f"Error encoding image: {e}")
            return "Error processing image."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    }
                ]
            }
        ]
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 256, 
            "temperature": 0.0, 
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error during Fireworks inference: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return "Error during inference."


def get_model(model_id: str, device: str = 'cuda') -> BaseVLM:
    """Factory function to instantiate the appropriate VLM handler."""
    model_lower = model_id.lower()
    
    if "fireworks" in model_lower or "32b" in model_lower:
        return FireworksQwenVLM(model_id, device)
    elif 'qwen' in model_lower:
        return QwenVLM(model_id, device)
    elif 'smol' in model_lower or 'idefics' in model_lower:
        return SmolVLM(model_id, device)
    else:
        raise ValueError(
            f"Unsupported model: {model_id}. "
            f"Supported: Qwen, SmolVLM, Fireworks"
        )
