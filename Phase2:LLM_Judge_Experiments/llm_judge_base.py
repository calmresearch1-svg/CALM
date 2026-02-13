import os
import time
import json
import re
from typing import Optional, Dict, List, Any, Union
from abc import ABC, abstractmethod
from PIL import Image
from pathlib import Path

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    genai = None

try:
    import openai
except ImportError:
    openai = None


class LLM_Judge(ABC):
    """
    Base class for LLM Judges.
    """

    def __init__(self, aspect_instruction: str = "", output_format: str = ""):
        self.aspect_instruction = aspect_instruction
        self.output_format = output_format

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Loads an image from the given path."""
        try:
            if os.path.exists(image_path):
                return Image.open(image_path)
            else:
                print(f"Warning: Image not found: {image_path}")
                return None
        except Exception as e:
            print(f"Warning: Error loading image {image_path}: {e}")
            return None

    def construct_single_prompt(self, question: str, response: str) -> str:
        return f"""**[System Role]**
            {self.aspect_instruction}

            **Input Data:**
            *   **Question:** {question}
            *   **Response (Original):** {response}

            **Output Format:**
            {self.output_format}
        """

    def construct_combined_prompt(self, original_q: str, original_a: str, cf_q: str, cf_a: str) -> str:
        return f"""**[System Role]**
            {self.aspect_instruction}

            **Input Data:**
            *   **Original Question:** {original_q}
            *   **Model Response (Original):** {original_a}
            *   **Counterfactual Question:** {cf_q}
            *   **Model Response (Counterfactual):** {cf_a}

            **Output Format:**
            {self.output_format}
        """

    def evaluate_single_question(self, question: str, response: str, image_path: Optional[str] = None, image_required: bool = False) -> str:
        """
        Evaluates a single question-response pair.
        Returns the raw response string from the LLM.
        """
        prompt = self.construct_single_prompt(question, response)
        img = None
        if image_required and image_path:
            img = self.load_image(image_path)
        
        return self.generate_response(prompt, images=img)

    def evaluate_combined_question(self, original_q: str, original_resp: str, cf_q: str, cf_resp: str, image_path: Optional[str] = None, image_required: bool = False) -> str:
        """
        Evaluates an original and counterfactual question-response pair together.
        Returns the raw response string from the LLM.
        """
        prompt = self.construct_combined_prompt(original_q, original_resp, cf_q, cf_resp)
        img = None
        if image_required and image_path:
            img = self.load_image(image_path)

        return self.generate_response(prompt, images=img)

    @abstractmethod
    def generate_response(self, prompt: str, images: Union[Image.Image, List[Image.Image], None] = None) -> str:
        """Abstract method to generate response from the specific LLM implementation."""
        pass


class Gemini_Evaluator(LLM_Judge):
    """
    Gemini implementation of the LLM Judge.
    """

    def __init__(self, api_key: str, aspect_instruction: str = "", output_format: str = "", model_name: str = "gemini-2.0-flash", rate_limit_delay: float = 1.0):
        super().__init__(aspect_instruction, output_format)
        self.model_name = model_name
        self.rate_limit_delay = rate_limit_delay
        
        if genai:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        else:
            self.model = None
            self.safety_settings = None
            print("Warning: google-generativeai not installed. Using dummy model.")

    def generate_response(self, prompt: str, images: Union[Image.Image, List[Image.Image], None] = None) -> str:
        if not self.model:
            return '{"reasoning": "Gemini not initialized or installed", "score": 0}'

        content = [prompt]
        if images:
            if isinstance(images, list):
                content.extend(images)
            else:
                content.append(images)

        for attempt in range(3):
            try:
                response = self.model.generate_content(
                    content,
                    safety_settings=self.safety_settings
                )
                time.sleep(self.rate_limit_delay)
                return response.text
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                time.sleep((attempt + 1) * 2)
        
        return '{"reasoning": "Failed to generate response after retries", "score": 0}'


class OpenAI_Evaluator(LLM_Judge):
    """
    OpenAI implementation of the LLM Judge.
    """

    def __init__(self, api_key: str, aspect_instruction: str = "", output_format: str = "", model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        super().__init__(aspect_instruction, output_format)
        self.model_name = model_name
        self.temperature = temperature
        
        if openai:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = None
            print("Warning: openai not installed. Using dummy client.")

    def generate_response(self, prompt: str, images: Union[Image.Image, List[Image.Image], None] = None) -> str:
        if not self.client:
            return '{"reasoning": "OpenAI not initialized or installed", "score": 0}'

        # OpenAI handling of images usually involves passing image URLs or base64. 
        # For simplicity in this base implementation, we will check if images are passed and warn if not implemented fully for local images without base64 conversion.
        
        
        messages = [
            {"role": "system", "content": self.aspect_instruction if self.aspect_instruction else "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # If images are provided, we would need to encode them. 
        # For now, let's assume text-only or implement simple base64 if needed later. 
        # The prompt passed to generate_response usually mimics the user message.
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f'{{"reasoning": "API Error: {str(e)}", "score": 0}}'
 


def extract_score(text: str) -> Optional[int]:
    # Try parsing as JSON first
    try:
        # cleanup markdown code blocks if present
        clean_text = text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        
        data = json.loads(clean_text)
        return int(data.get("score"))
    except Exception:
        pass

    # Look for "Score: X" or just the number at the end (fallback)
    match = re.search(r'Score:\s*(\d)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
        
    # Regex for json-like pattern "score": 3
    match = re.search(r'"score":\s*(\d)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None