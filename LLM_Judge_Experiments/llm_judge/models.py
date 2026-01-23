"""
Model factory for creating LangChain LLM instances.
Supports Gemini, OpenAI, NVIDIA, and Fal.ai models with a unified interface.
"""

import os
from typing import Optional, List, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage

from llm_judge.config import get_api_key


# NVIDIA model prefixes for detection (includes vendor prefixes)
NVIDIA_MODEL_PREFIXES = ("llama", "qwen", "qwen/", "nvidia", "nvidia/", "mistral", "deepseek", "meta/", "openai/gpt-oss")

# Specific NVIDIA models that don't follow prefix patterns
NVIDIA_SPECIFIC_MODELS = (
    "openai/gpt-oss",  # NVIDIA-hosted OpenAI models
)


def get_model_provider(model_name: str) -> str:
    """Determine the provider based on model name."""
    model_lower = model_name.lower()
    
    # Check specific NVIDIA models first (handles gpt-oss-* etc.)
    if any(model_lower.startswith(m) for m in NVIDIA_SPECIFIC_MODELS):
        return "nvidia"
    
    # Then check by prefix
    if model_lower.startswith("gemini"):
        return "gemini"
    elif model_lower.startswith("gpt"):
        return "openai"
    elif model_name.startswith("fal:"):
        return "fal"
    elif any(model_lower.startswith(prefix) for prefix in NVIDIA_MODEL_PREFIXES):
        return "nvidia"
    else:
        raise ValueError(
            f"Unknown model provider for: {model_name}. "
            f"Supported: gemini*, gpt*, fal:*, llama*, qwen*, nvidia/*, mistral*, deepseek*, gpt-oss-*"
        )


class ModelFactory:
    """Factory for creating LangChain LLM instances."""
    
    @staticmethod
    def create(
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> BaseChatModel:
        """
        Create a LangChain chat model instance.
        
        Args:
            model_name: Name of the model (e.g., "gemini-2.0-flash", "gpt-4o-mini", "meta/llama-3.2-90b-vision-instruct")
            api_key: Optional API key (will use env var if not provided)
            temperature: Generation temperature
            **kwargs: Additional model-specific arguments
            
        Returns:
            A LangChain BaseChatModel instance
        """
        provider = get_model_provider(model_name)
        
        if provider == "gemini":
            return ModelFactory._create_gemini(model_name, api_key, temperature, **kwargs)
        elif provider == "openai":
            return ModelFactory._create_openai(model_name, api_key, temperature, **kwargs)
        elif provider == "nvidia":
            return ModelFactory._create_nvidia(model_name, api_key, temperature, **kwargs)
        elif provider == "fal":
            return ModelFactory._create_fal(model_name, api_key, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def _create_gemini(
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> BaseChatModel:
        """Create a Gemini model instance."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Gemini models. "
                "Install with: pip install langchain-google-genai"
            )
        
        key = api_key or get_api_key("gemini")
        if not key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY in environment or .env file."
            )
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=key,
            temperature=temperature,
            **kwargs
        )
    
    @staticmethod
    def _create_openai(
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> BaseChatModel:
        """Create an OpenAI model instance."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI models. "
                "Install with: pip install langchain-openai"
            )
        
        key = api_key or get_api_key("openai")
        if not key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in environment or .env file."
            )
        
        return ChatOpenAI(
            model=model_name,
            api_key=key,
            temperature=temperature,
            **kwargs
        )
    
    @staticmethod
    def _create_fal(
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> BaseChatModel:
        """Create a Fal.ai model instance."""
        try:
            import fal_client
        except ImportError:
            raise ImportError(
                "fal-client is required for Fal.ai models. "
                "Install with: pip install fal-client"
            )
        
        key = api_key or get_api_key("fal")
        if not key:
            raise ValueError(
                "Fal.ai API key not found. Set FAL_KEY in environment or .env file."
            )
        
        # Set the API key in environment for fal_client
        os.environ["FAL_KEY"] = key
        
        # Extract actual model name (remove "fal:" prefix)
        actual_model = model_name[4:]  # Remove "fal:" prefix
        
        return FalChatModel(
            model=actual_model,
            temperature=temperature,
            **kwargs
        )
    
    @staticmethod
    def _create_nvidia(
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> BaseChatModel:
        """Create an NVIDIA NIM model instance."""
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError:
            raise ImportError(
                "langchain-nvidia-ai-endpoints is required for NVIDIA models. "
                "Install with: pip install langchain-nvidia-ai-endpoints"
            )
        
        key = api_key or get_api_key("nvidia")
        if not key:
            raise ValueError(
                "NVIDIA API key not found. Set NVIDIA_API_KEY in environment or .env file."
            )
        
        return ChatNVIDIA(
            model=model_name,
            api_key=key,
            temperature=temperature,
            **kwargs
        )


class FalChatModel(BaseChatModel):
    """
    LangChain-compatible wrapper for Fal.ai's any-llm endpoint.
    Supports both text-only and vision (image+text) inputs.
    
    Usage:
        model = FalChatModel(model="google/gemini-2.5-flash")
        response = model.invoke([HumanMessage(content="Hello")])
    """
    
    model: str = "google/gemini-2.5-flash-lite"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    
    @property
    def _llm_type(self) -> str:
        return "fal-any-llm"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Generate a response using Fal.ai's any-llm endpoint."""
        import fal_client
        from langchain_core.outputs import ChatResult, ChatGeneration
        
        # Convert messages to a single prompt string
        # Also extract any image URLs for vision endpoint
        prompt_parts = []
        system_prompt = None
        image_url = None
        
        for msg in messages:
            content = msg.content
            # Handle multimodal content (list of dicts with type/text/image_url)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Extract image URL for vision endpoint
                            url_data = item.get("image_url", {})
                            if isinstance(url_data, dict):
                                image_url = url_data.get("url")
                            else:
                                image_url = url_data
                    else:
                        text_parts.append(str(item))
                content = "\n".join(text_parts)
            
            if msg.type == "system":
                system_prompt = content
            else:
                prompt_parts.append(content)
        
        prompt = "\n".join(prompt_parts)
        
        # Build request arguments
        arguments = {
            "prompt": prompt,
            "model": self.model,
        }
        
        if system_prompt:
            arguments["system_prompt"] = system_prompt
        
        if self.temperature is not None:
            arguments["temperature"] = self.temperature
        
        if self.max_tokens is not None:
            arguments["max_tokens"] = self.max_tokens
        
        # Choose endpoint based on whether we have an image
        if image_url:
            arguments["image_url"] = image_url
            endpoint = "fal-ai/any-llm/vision"
        else:
            endpoint = "fal-ai/any-llm"
        
        # Call Fal.ai API
        result = fal_client.subscribe(
            endpoint,
            arguments=arguments,
        )
        
        output_text = result.get("output", "")
        
        # Return in LangChain format
        message = AIMessage(content=output_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model, "temperature": self.temperature}

