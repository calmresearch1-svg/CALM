"""
Base LLM Judge class for evaluating model responses.
Provides both sync and async evaluation interfaces.
"""

import asyncio
import time
import re
from pathlib import Path
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from llm_judge.config import AspectConfig, DEFAULT_OUTPUT_FORMAT
from llm_judge.models import ModelFactory
from llm_judge.prompts import PromptBuilder
from llm_judge.parsers import extract_score
from llm_judge.utils import build_multimodal_content


@dataclass
class EvalTask:
    """Evaluation task for combined (original + counterfactual) analysis."""
    entry_id: str
    image_path: str
    image_path_relative: str
    original_q: str
    original_a: str
    cf_q: str
    cf_a: str
    model_name: str


@dataclass
class SingleEvalTask:
    """Evaluation task for single question analysis."""
    entry_id: str
    image_path: str
    image_path_relative: str
    question: str
    response: str
    model_name: str


class LLMJudge:
    """
    Simple LLM-as-a-Judge evaluator.
    
    This class provides a clean interface for evaluating model responses.
    All file handling should be done in the aspect-specific scripts.
    
    Usage:
        judge = LLMJudge(aspect_config, judge_model="gemini-2.5-flash")
        result = judge.evaluate_combined(orig_q, orig_a, cf_q, cf_a, image_path)
        # result = {"raw_response": "...", "score": 5}
    """
    
    def __init__(
        self,
        aspect_config: AspectConfig,
        judge_model: str = "gemini-2.0-flash",
        evidence_modality: str = "medical image",
        api_key: Optional[str] = None,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
    ):
        """
        Initialize the LLM Judge.
        
        Args:
            aspect_config: Configuration for the aspect being evaluated
            judge_model: Name of the judge model to use
            evidence_modality: Dataset-specific evidence type (e.g., "chest X-ray image")
            api_key: Optional API key (uses env var if not provided)
            rate_limit_delay: Delay between API calls in seconds
            max_retries: Maximum number of retries for failed API calls
        """
        self.aspect_config = aspect_config
        self.judge_model = judge_model
        self.evidence_modality = evidence_modality
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        
        # Initialize model
        self.model = ModelFactory.create(judge_model, api_key=api_key)
        
        # Initialize prompt builder with aspect instruction
        self.prompt_builder = PromptBuilder(
            aspect_instruction=aspect_config.get_instruction(evidence_modality),
            output_format=DEFAULT_OUTPUT_FORMAT
        )
    
    def evaluate_single(
        self,
        question: str,
        response: str,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single question-response pair.
        
        Args:
            question: The question that was asked
            response: The model's response to evaluate
            image_path: Optional path to associated image
            
        Returns:
            {"raw_response": str, "score": int or None}
        """
        prompt = self.prompt_builder.build_single_prompt(question, response)
        return self._call_judge(prompt, image_path)
    
    def evaluate_combined(
        self,
        original_q: str,
        original_a: str,
        cf_q: str,
        cf_a: str,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate original and counterfactual question-response pairs.
        
        Args:
            original_q: Original question
            original_a: Model's response to original question
            cf_q: Counterfactual question
            cf_a: Model's response to counterfactual question
            image_path: Optional path to associated image
            
        Returns:
            {"raw_response": str, "score": int or None}
        """
        prompt = self.prompt_builder.build_combined_prompt(
            original_q, original_a, cf_q, cf_a
        )
        return self._call_judge(prompt, image_path)
    
    def _call_judge(
        self,
        prompt: str,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make the API call to the judge model.
        
        Args:
            prompt: The evaluation prompt
            image_path: Optional path to image
            
        Returns:
            {"raw_response": str, "score": int or None}
        """
        # Build content with optional image
        include_image = self.aspect_config.requires_image and image_path
        content = build_multimodal_content(prompt, image_path if include_image else None)
        
        message = HumanMessage(content=content)
        
        for attempt in range(self.max_retries):
            try:
                response = self.model.invoke([message])
                time.sleep(self.rate_limit_delay)
                
                raw_text = response.content if hasattr(response, 'content') else str(response)
                score = extract_score(raw_text)
                
                return {
                    "raw_response": raw_text,
                    "score": score
                }
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                
                # Check for rate limit and extract delay
                retry_match = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", str(e))
                if retry_match:
                    delay = int(retry_match.group(1)) + 1
                    print(f"Rate limit hit. Waiting {delay} seconds...")
                    time.sleep(delay)
                else:
                    delay = (attempt + 1) * 2
                    print(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
        
        return {
            "raw_response": '{"reasoning": "Failed after retries", "score": 0}',
            "score": None
        }


class AsyncLLMJudge:
    """
    Async LLM Judge for concurrent evaluation.
    
    Uses asyncio to process multiple evaluations concurrently,
    with semaphore-based rate limiting and incremental file writing.
    
    Usage:
        judge = AsyncLLMJudge(aspect_config, concurrency=10)
        await judge.evaluate_batch(tasks, output_file)
    """
    
    def __init__(
        self,
        aspect_config: AspectConfig,
        judge_model: str = "gemini-2.5-flash",
        evidence_modality: str = "medical image",
        concurrency: int = 10,
        max_retries: int = 8,
    ):
        self.aspect_config = aspect_config
        self.judge_model = judge_model
        self.evidence_modality = evidence_modality
        self.concurrency = concurrency
        self.max_retries = max_retries
        
        # Initialize model
        self.model = ModelFactory.create(judge_model)
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(
            aspect_instruction=aspect_config.get_instruction(evidence_modality),
            output_format=DEFAULT_OUTPUT_FORMAT
        )
        
        # Semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(concurrency)
    
    async def evaluate_single_task_with_write(
        self, 
        task: EvalTask,
        output_file: Path,
        file_lock: asyncio.Lock,
        pbar,
    ) -> Dict[str, Any]:
        """Evaluate a single task and write result immediately."""
        async with self.semaphore:
            try:
                result = await self._call_judge_async(task)
            except Exception as e:
                print(f"\nError on entry {task.entry_id}: {e}")
                result = {"raw_response": f"Error: {str(e)}", "score": None}
            
            # Build output entry
            output_entry = {
                "entry_id": task.entry_id,
                "image_path": task.image_path_relative,
                "llm_judge_score": result["score"],
                "llm_judge_raw_response": result["raw_response"],
                "original_question": task.original_q,
                "original_response": task.original_a,
                "counterfactual_question": task.cf_q,
                "counterfactual_response": task.cf_a,
                "judge_model": self.judge_model,
                "evaluated_model": task.model_name,
                "aspect": self.aspect_config.name,
            }
            
            # Write immediately with lock
            async with file_lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(output_entry) + "\n")
            
            pbar.update(1)
            return output_entry
    
    async def _call_judge_async(self, task: EvalTask) -> Dict[str, Any]:
        """Make async API call to judge model."""
        # Build prompt
        prompt = self.prompt_builder.build_combined_prompt(
            task.original_q, task.original_a, task.cf_q, task.cf_a
        )
        
        # Build content with image
        include_image = self.aspect_config.requires_image and task.image_path
        content = build_multimodal_content(prompt, task.image_path if include_image else None)
        message = HumanMessage(content=content)
        
        for attempt in range(self.max_retries):
            try:
                # Use asyncio.to_thread for sync model.invoke
                response = await asyncio.to_thread(self.model.invoke, [message])
                
                raw_text = response.content if hasattr(response, 'content') else str(response)
                score = extract_score(raw_text)
                
                return {"raw_response": raw_text, "score": score}
                
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate limit" in error_str.lower()
                
                # Check for explicit retry_delay in error (Gemini style)
                retry_match = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", error_str)
                
                if retry_match:
                    # Use the API's suggested delay
                    delay = int(retry_match.group(1)) + 1
                elif is_rate_limit:
                    # Exponential backoff for rate limits: 4, 8, 16, 32, 64, 120...
                    delay = min(2 ** (attempt + 2), 120)
                else:
                    # Standard exponential backoff for other errors
                    delay = min(2 ** attempt, 30)
                
                print(f"\n  Attempt {attempt + 1}/{self.max_retries} failed for {task.entry_id}: {error_str[:80]}")
                print(f"  {'Rate limit' if is_rate_limit else 'Error'}. Waiting {delay}s before retry...")
                await asyncio.sleep(delay)
        
        return {"raw_response": '{"reasoning": "Failed after all retries", "score": 0}', "score": None}
    
    async def evaluate_batch(
        self,
        tasks: List[EvalTask],
        output_file: Path,
    ) -> List[Dict[str, Any]]:
        """Evaluate combined tasks concurrently, writing results as they complete."""
        from tqdm.asyncio import tqdm_asyncio
        
        print(f"Processing {len(tasks)} combined entries with concurrency={self.concurrency}...")
        print(f"Results will be written incrementally to: {output_file}")
        
        file_lock = asyncio.Lock()
        results = []
        
        with tqdm_asyncio(total=len(tasks), desc="Evaluating") as pbar:
            # Create all tasks with incremental writing
            coroutines = [
                self.evaluate_single_task_with_write(task, output_file, file_lock, pbar)
                for task in tasks
            ]
            
            # Use as_completed to process as they finish
            for coro in asyncio.as_completed(coroutines):
                try:
                    result = await coro
                    results.append(result)
                except Exception as e:
                    print(f"\nTask failed: {e}")
        
        return results
    
    # ========== Single Question Evaluation Methods ==========
    
    async def evaluate_single_with_write(
        self, 
        task: 'SingleEvalTask',
        output_file: Path,
        file_lock: asyncio.Lock,
        pbar,
    ) -> Dict[str, Any]:
        """Evaluate a single Q&A task and write result immediately."""
        async with self.semaphore:
            try:
                result = await self._call_judge_async_single(task)
            except Exception as e:
                print(f"\nError on entry {task.entry_id}: {e}")
                result = {"raw_response": f"Error: {str(e)}", "score": None}
            
            # Build output entry
            output_entry = {
                "entry_id": task.entry_id,
                "image_path": task.image_path_relative,
                "llm_judge_score": result["score"],
                "llm_judge_raw_response": result["raw_response"],
                "question": task.question,
                "response": task.response,
                "judge_model": self.judge_model,
                "evaluated_model": task.model_name,
                "aspect": self.aspect_config.name,
            }
            
            # Write immediately with lock
            async with file_lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(output_entry) + "\n")
            
            pbar.update(1)
            return output_entry
    
    async def _call_judge_async_single(self, task: 'SingleEvalTask') -> Dict[str, Any]:
        """Make async API call for single Q&A evaluation."""
        # Build single prompt
        prompt = self.prompt_builder.build_single_prompt(task.question, task.response)
        
        # Build content with image
        include_image = self.aspect_config.requires_image and task.image_path
        content = build_multimodal_content(prompt, task.image_path if include_image else None)
        message = HumanMessage(content=content)
        
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(self.model.invoke, [message])
                
                raw_text = response.content if hasattr(response, 'content') else str(response)
                score = extract_score(raw_text)
                
                return {"raw_response": raw_text, "score": score}
                
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate limit" in error_str.lower()
                
                retry_match = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", error_str)
                
                if retry_match:
                    delay = int(retry_match.group(1)) + 1
                elif is_rate_limit:
                    delay = min(2 ** (attempt + 2), 120)
                else:
                    delay = min(2 ** attempt, 30)
                
                print(f"\n  Attempt {attempt + 1}/{self.max_retries} failed for {task.entry_id}: {error_str[:80]}")
                print(f"  {'Rate limit' if is_rate_limit else 'Error'}. Waiting {delay}s before retry...")
                await asyncio.sleep(delay)
        
        return {"raw_response": '{"reasoning": "Failed after all retries", "score": 0}', "score": None}
    
    async def evaluate_single_batch(
        self,
        tasks: List['SingleEvalTask'],
        output_file: Path,
    ) -> List[Dict[str, Any]]:
        """Evaluate single Q&A tasks concurrently, writing results as they complete."""
        from tqdm.asyncio import tqdm_asyncio
        
        print(f"Processing {len(tasks)} single entries with concurrency={self.concurrency}...")
        print(f"Results will be written incrementally to: {output_file}")
        
        file_lock = asyncio.Lock()
        results = []
        
        with tqdm_asyncio(total=len(tasks), desc="Evaluating") as pbar:
            coroutines = [
                self.evaluate_single_with_write(task, output_file, file_lock, pbar)
                for task in tasks
            ]
            
            for coro in asyncio.as_completed(coroutines):
                try:
                    result = await coro
                    results.append(result)
                except Exception as e:
                    print(f"\nTask failed: {e}")
        
        return results
