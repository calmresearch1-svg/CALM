"""
Data handlers for different datasets.
Each handler knows how to extract fields from its specific JSONL format.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path

from llm_judge.config import DATASET_IMAGE_ROOTS


class BaseDataHandler(ABC):
    """Base class for dataset-specific data handlers."""
    
    def __init__(self, dataset_name: str, image_root: Optional[Path] = None):
        """
        Initialize the data handler.
        
        Args:
            dataset_name: Name of the dataset
            image_root: Optional override for image root directory
        """
        self.dataset_name = dataset_name
        self.image_root = image_root or DATASET_IMAGE_ROOTS.get(dataset_name)
        
        if self.image_root is None:
            raise ValueError(f"Unknown dataset: {dataset_name}. Add it to DATASET_IMAGE_ROOTS in config.py")
    
    @abstractmethod
    def get_entry_id(self, entry: Dict[str, Any]) -> str:
        """Get unique identifier for an entry."""
        pass
    
    @abstractmethod
    def get_question(self, entry: Dict[str, Any]) -> str:
        """Get the question text from an entry."""
        pass
    
    @abstractmethod
    def get_response(self, entry: Dict[str, Any]) -> str:
        """Get the model response from an entry."""
        pass
    
    @abstractmethod
    def get_image_path(self, entry: Dict[str, Any]) -> Optional[str]:
        """Get the relative image path from an entry."""
        pass
    
    def get_full_image_path(self, entry: Dict[str, Any]) -> Optional[str]:
        """Get the full absolute image path."""
        rel_path = self.get_image_path(entry)
        if rel_path is None:
            return None
        
        full_path = self.image_root / rel_path
        if full_path.exists():
            return str(full_path)
        
        # Try without any prefix manipulation
        if Path(rel_path).exists():
            return rel_path
            
        print(f"Warning: Image not found: {full_path}")
        return None
    
    def get_model_name(self, entry: Dict[str, Any]) -> Optional[str]:
        """Get the evaluated model name if available."""
        return entry.get("model_name")
    
    def get_counterfactual_question(self, entry: Dict[str, Any]) -> Optional[str]:
        """Get counterfactual question if present in entry (for merged files)."""
        return entry.get("counterfactual_question") or entry.get("cf_question")
    
    def get_counterfactual_response(self, entry: Dict[str, Any]) -> Optional[str]:
        """Get counterfactual response if present in entry (for merged files)."""
        return entry.get("counterfactual_response") or entry.get("cf_response")


class IUXrayHandler(BaseDataHandler):
    """Handler for IU X-Ray dataset JSONL files."""
    
    def __init__(self, image_root: Optional[Path] = None):
        super().__init__("iu_xray", image_root)
    
    def get_entry_id(self, entry: Dict[str, Any]) -> str:
        return str(entry.get("question_index", entry.get("id", "")))
    
    def get_question(self, entry: Dict[str, Any]) -> str:
        return entry.get("question", "")
    
    def get_response(self, entry: Dict[str, Any]) -> str:
        return entry.get("response", "")
    
    def get_image_path(self, entry: Dict[str, Any]) -> Optional[str]:
        return entry.get("image_path")
    
    def get_original_question(self, entry: Dict[str, Any]) -> Optional[str]:
        """For CF files, get the original question field."""
        return entry.get("original_question")


class HAM10000Handler(BaseDataHandler):
    """Handler for HAM10000 dataset JSONL files."""
    
    def __init__(self, image_root: Optional[Path] = None):
        super().__init__("ham10000", image_root)
    
    def get_entry_id(self, entry: Dict[str, Any]) -> str:
        return str(entry.get("question_id", entry.get("id", "")))
    
    def get_question(self, entry: Dict[str, Any]) -> str:
        """Get question with options appended for HAM10000."""
        question = entry.get("question", "")
        options = entry.get("options", "")
        if options:
            return f"{question}\nOptions: {options}"
        return question
    
    def get_question_only(self, entry: Dict[str, Any]) -> str:
        """Get just the question text without options."""
        return entry.get("question", "")
    
    def get_options(self, entry: Dict[str, Any]) -> str:
        """Get options text."""
        return entry.get("options", "")
    
    def get_response(self, entry: Dict[str, Any]) -> str:
        return entry.get("response", "")
    
    def get_image_path(self, entry: Dict[str, Any]) -> Optional[str]:
        # HAM10000 includes folder prefix like "ham10000_testset/ISIC_xxx.jpg"
        return entry.get("image")


class HarvardFairVLMedHandler(BaseDataHandler):
    """Handler for Harvard FairVLMed dataset JSONL files."""
    
    def __init__(self, image_root: Optional[Path] = None):
        super().__init__("harvard_fairvlmed", image_root)
    
    def get_entry_id(self, entry: Dict[str, Any]) -> str:
        # Use question_id from fairness data
        return str(entry.get("question_id", entry.get("question_index", entry.get("id", ""))))
    
    def get_question(self, entry: Dict[str, Any]) -> str:
        # Fairness data uses 'text' field for question
        return entry.get("text", entry.get("question", ""))
    
    def get_response(self, entry: Dict[str, Any]) -> str:
        # Fairness data uses 'model_output' field
        return entry.get("model_output", entry.get("response", entry.get("answer", "")))
    
    def get_image_path(self, entry: Dict[str, Any]) -> Optional[str]:
        return entry.get("image", entry.get("image_path"))


def get_data_handler(dataset_name: str, image_root: Optional[Path] = None) -> BaseDataHandler:
    """
    Factory function to get the appropriate data handler.
    
    Args:
        dataset_name: Name of the dataset ("iu_xray", "ham10000", "harvard_fairvlmed")
        image_root: Optional override for image root directory
        
    Returns:
        Appropriate BaseDataHandler subclass instance
    """
    handlers = {
        "iu_xray": IUXrayHandler,
        "ham10000": HAM10000Handler,
        "harvard_fairvlmed": HarvardFairVLMedHandler,
    }
    
    handler_class = handlers.get(dataset_name.lower())
    if handler_class is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(handlers.keys())}"
        )
    
    return handler_class(image_root)
