"""JSON response parsing utilities."""

import json
import re
from typing import Dict, Any, Optional


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from model response string.

    Handles cases where response might contain extra text or whitespace.

    Args:
        response: Raw response string from model

    Returns:
        Parsed JSON dictionary or None if parsing fails
    """
    if not response:
        return None

    # Try to find JSON object in response
    # Look for { ... } pattern
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(0)
    else:
        # If no match, try the whole string
        json_str = response.strip()

    # Try to parse JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try cleaning the string
        json_str = json_str.strip()
        
        # Remove leading/trailing non-JSON characters
        json_str = re.sub(r'^[^{]*', '', json_str)
        json_str = re.sub(r'[^}]*$', '', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Last resort: try to extract question and answer manually
            question_match = re.search(r'"question"\s*:\s*"([^"]+)"', json_str)
            answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', json_str)
            
            if question_match and answer_match:
                return {
                    "question": question_match.group(1),
                    "answer": answer_match.group(1),
                }
            
            return None

