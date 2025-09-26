import re
from typing import Any, Dict, List
from mathruler.grader import extract_boxed_content, grade_answer

def format_reward(response: str) -> float:
    """
    Ensure the response strictly contains only `\boxed{}` content.
    If there are any extra characters outside the boxed content, score is 0.
    If the format matches exactly, score is 1.0.
    """
    # Pattern to strictly match a single `\boxed{}` expression in the entire response
    pattern = re.compile(r"^\s*\\boxed\{[^}]+\}\s*$") ## no think
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Extract the content inside the `\boxed{}` and compare it against the ground truth.
    If they match exactly, the accuracy score is 1.0; otherwise, it's 0.0.
    """
    answer = extract_boxed_content(response)  # Extract the content inside \boxed{}
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    """
    Compute the overall score for response evaluation. 
    Combines format correctness and answer accuracy.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    
    scores = []
    for reward_input in reward_inputs:
        # Normalize and sanitize the response (handle specific formatting cases if necessary)
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format

        # Compute format correctness
        format_score = format_reward(response)

        # Compute accuracy correctness
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])

        # Combine format and accuracy scores using weights
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "response_length": reward_input["response_length"]  # Include response length for debugging
            }
        )
    return scores