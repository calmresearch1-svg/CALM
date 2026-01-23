"""
Utility functions for LLM Judge framework.
Includes statistics calculation, image encoding, and file I/O helpers.
"""

import os
import json
import base64
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


def calculate_statistics(filepath: str) -> Dict[str, Any]:
    """
    Calculate mean and standard deviation of scores from output JSONL file.
    
    Args:
        filepath: Path to the output JSONL file
        
    Returns:
        Dictionary with 'mean', 'std', 'count', 'valid_count', 'scores' keys
    """
    scores = []
    total_count = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                total_count += 1
                try:
                    entry = json.loads(line)
                    score = entry.get("llm_judge_score")
                    if score is not None and isinstance(score, (int, float)) and 1 <= score <= 5:
                        scores.append(score)
                except json.JSONDecodeError:
                    continue
    
    if not scores:
        return {
            "mean": None,
            "std": None,
            "count": total_count,
            "valid_count": 0,
            "scores": []
        }
    
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "count": total_count,
        "valid_count": len(scores),
        "scores": scores
    }


def print_statistics(filepath: str) -> None:
    """
    Calculate and print statistics from output JSONL file.
    
    Args:
        filepath: Path to the output JSONL file
    """
    stats = calculate_statistics(filepath)
    
    print("\n" + "=" * 50)
    print("LLM JUDGE EVALUATION STATISTICS")
    print("=" * 50)
    print(f"File: {filepath}")
    print(f"Total entries: {stats['count']}")
    print(f"Valid scores: {stats['valid_count']}")
    
    if stats['mean'] is not None:
        print(f"Mean score: {stats['mean']:.4f}")
        print(f"Std deviation: {stats['std']:.4f}")
    else:
        print("No valid scores found.")
    
    print("=" * 50 + "\n")


def encode_image_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string or None if file doesn't exist
    """
    if not os.path.exists(image_path):
        return None
    
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """
    Get the MIME type for an image based on file extension.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        MIME type string (e.g., "image/jpeg")
    """
    ext = image_path.lower().split(".")[-1]
    mime_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }
    return mime_types.get(ext, "image/png")


def build_multimodal_content(text: str, image_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Build multimodal content list for LangChain messages.
    
    Args:
        text: Text content
        image_path: Optional path to an image
        
    Returns:
        List of content dictionaries compatible with LangChain
    """
    content = [{"type": "text", "text": text}]
    
    if image_path and os.path.exists(image_path):
        mime_type = get_image_mime_type(image_path)
        base64_img = encode_image_base64(image_path)
        
        if base64_img:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}
            })
    
    return content


def generate_output_filename(
    input_filename: str,
    judge_model: str,
    aspect: str = ""
) -> str:
    """
    Generate output filename following the naming convention.
    
    Args:
        input_filename: Original input filename (without path)
        judge_model: Name of the judge model used
        aspect: Optional aspect name (if not in input filename)
        
    Returns:
        Generated output filename
    """
    # Remove extension from input
    base_name = Path(input_filename).stem
    
    # Clean up model name for filename
    model_clean = judge_model.replace("/", "-").replace(".", "-")
    
    # Get current date
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    return f"{base_name}_LLJ_result_{model_clean}_{date_str}.jsonl"


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load entries from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
    return entries


def save_jsonl_entry(filepath: str, entry: Dict[str, Any]) -> None:
    """
    Append a single entry to a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        entry: Dictionary to append
    """
    with open(filepath, 'a') as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
