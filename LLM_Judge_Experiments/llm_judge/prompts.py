"""
Prompt templates for LLM Judge evaluations.
Provides single and combined evaluation prompt constructors.
"""

from langchain_core.prompts import PromptTemplate
from llm_judge.config import DEFAULT_OUTPUT_FORMAT


# Single evaluation template (for privacy, safety)
SINGLE_EVAL_TEMPLATE = PromptTemplate(
    input_variables=["aspect_instruction", "output_format", "question", "response"],
    template="""[System Role]
{aspect_instruction}

[Input Data]
- Question: {question}
- Model Response: {response}

[Output Format]
{output_format}
"""
)


# Combined evaluation template (for robustness, fairness)
COMBINED_EVAL_TEMPLATE = PromptTemplate(
    input_variables=[
        "aspect_instruction", "output_format", 
        "original_q", "original_a", "cf_q", "cf_a"
    ],
    template="""[System Role]
{aspect_instruction}

[Input Data]
- Original Question: {original_q}
- Model Response (Original): {original_a}
- Counterfactual Question: {cf_q}
- Model Response (Counterfactual): {cf_a}

[Output Format]
{output_format}
"""
)


class PromptBuilder:
    """Builds prompts for LLM Judge evaluations."""
    
    def __init__(self, aspect_instruction: str, output_format: str = DEFAULT_OUTPUT_FORMAT):
        """
        Initialize the prompt builder.
        
        Args:
            aspect_instruction: The instruction text for the specific aspect
            output_format: Expected output format string
        """
        self.aspect_instruction = aspect_instruction
        self.output_format = output_format
    
    def build_single_prompt(self, question: str, response: str) -> str:
        """
        Build a prompt for single question evaluation.
        
        Args:
            question: The question that was asked
            response: The model's response to evaluate
            
        Returns:
            Formatted prompt string
        """
        return SINGLE_EVAL_TEMPLATE.format(
            aspect_instruction=self.aspect_instruction,
            output_format=self.output_format,
            question=question,
            response=response
        )
    
    def build_combined_prompt(
        self, 
        original_q: str, 
        original_a: str, 
        cf_q: str, 
        cf_a: str
    ) -> str:
        """
        Build a prompt for combined (original + counterfactual) evaluation.
        
        Args:
            original_q: Original question
            original_a: Model's response to original question
            cf_q: Counterfactual question
            cf_a: Model's response to counterfactual question
            
        Returns:
            Formatted prompt string
        """
        return COMBINED_EVAL_TEMPLATE.format(
            aspect_instruction=self.aspect_instruction,
            output_format=self.output_format,
            original_q=original_q,
            original_a=original_a,
            cf_q=cf_q,
            cf_a=cf_a
        )
