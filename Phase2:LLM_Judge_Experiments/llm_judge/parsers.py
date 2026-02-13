"""
Output parsing utilities for LLM Judge responses.
Handles JSON parsing with fallback regex extraction.
"""

import json
import re
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class JudgeOutput(BaseModel):
    """Expected output structure from LLM Judge."""
    reasoning: str = Field(description="Explanation of the score rationale")
    score: int = Field(description="Integer score from 1-5")


def extract_score(text: str) -> Optional[int]:
    """
    Extract score from LLM response text.
    
    Attempts JSON parsing first, then falls back to regex patterns.
    
    Args:
        text: Raw response text from LLM
        
    Returns:
        Extracted integer score (1-5) or None if extraction fails
    """
    # Try parsing as JSON first
    try:
        clean_text = _clean_json_response(text)
        data = json.loads(clean_text)
        score = int(data.get("score", 0))
        if 1 <= score <= 5:
            return score
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # Fallback: Look for "score": X pattern
    match = re.search(r'"score"\s*:\s*(\d)', text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return score
    
    # Fallback: Look for "Score: X" pattern
    match = re.search(r'Score:\s*(\d)', text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return score
    
    return None


def parse_judge_response(text: str) -> Dict[str, Any]:
    """
    Parse the full LLM Judge response.
    
    Args:
        text: Raw response text from LLM
        
    Returns:
        Dictionary with 'reasoning' and 'score' keys
    """
    try:
        clean_text = _clean_json_response(text)
        data = json.loads(clean_text)
        return {
            "reasoning": data.get("reasoning", ""),
            "score": int(data.get("score", 0))
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        # Return extracted score with original text as reasoning
        score = extract_score(text)
        return {
            "reasoning": text,
            "score": score if score else 0
        }


def _clean_json_response(text: str) -> str:
    """Clean markdown code blocks from JSON response."""
    clean_text = text.strip()
    
    # Remove markdown code block wrappers
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    
    return clean_text.strip()
