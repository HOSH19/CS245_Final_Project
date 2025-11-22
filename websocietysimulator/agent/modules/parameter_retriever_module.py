"""
ParameterRetriever module for extracting profile parameters from memory using LLM analysis.
"""

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger("websocietysimulator")


class ParameterRetriever:
    """
    Handles retrieval of profile parameters from memory using LLM analysis.
    """
    
    def __init__(self, memory=None, llm=None, profile_config=None):
        self.memory = memory
        self.llm = llm
        self.profile_config = profile_config or {}

    def retrieve_parameters(
        self, 
        steps: List[Dict[str, Any]], 
        profile_type: str
    ) -> List[str]:
        """Retrieve relevant parameter names from memory using LLM to extract best parameters."""
        if not self.llm or not self.memory:
            return self._get_default_parameters(profile_type)
        
        query = self._build_memory_query(steps, profile_type)
        memory_result = self.memory(query) if query else ""
        
        if not memory_result:
            return self._get_default_parameters(profile_type)
        
        parameters = self._extract_best_parameters_with_llm(memory_result, steps, profile_type)
        return parameters or self._get_default_parameters(profile_type)

    def _build_memory_query(self, steps: List[Dict[str, Any]], profile_type: str) -> str:
        """Build a query string from planner steps to search memory."""
        config = self.profile_config.get(profile_type, {})
        keywords = config.get("memory_query_keywords", [])
        related_descriptions = []
        
        for step in steps:
            description = step.get("description", "").lower()
            reasoning = step.get("reasoning", "").lower()
            
            if any(keyword in description for keyword in keywords):
                related_descriptions.append(description)
            if any(keyword in reasoning for keyword in keywords):
                related_descriptions.append(reasoning)
        
        return " ".join(related_descriptions[:3]).strip()

    def _extract_best_parameters_with_llm(
        self, 
        memory_result: str, 
        steps: List[Dict[str, Any]], 
        profile_type: str
    ) -> List[str]:
        """Use LLM to extract the best parameter names from memory content."""
        if not self.llm or not memory_result:
            return []
        
        config = self.profile_config.get(profile_type, {})
        profile_description = "user profile" if profile_type == "user" else "book/item profile"
        example_params = config.get("example_parameters", [])
        
        prompt = f"""Analyze planner steps and memory content to determine the best parameters for a {profile_description} schema.

Planner Steps:
{json.dumps(steps, default=str, indent=2)}

Memory Content:
{memory_result}

Example parameters: {', '.join(example_params)}

Return ONLY a JSON array of 3-5 most important parameter names (snake_case). Example: ["genre", "reading_level", "theme"]
"""
        
        try:
            response = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.2)
            return self._parse_llm_response(response)
        except Exception:
            return []

    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse and validate LLM response to extract parameter list."""
        response_clean = response.strip()
        
        if "```" in response_clean:
            json_match = re.search(r'\[.*?\]', response_clean, re.DOTALL)
            if json_match:
                response_clean = json_match.group(0)
        
        try:
            params = json.loads(response_clean)
            if isinstance(params, list):
                valid_params = []
                for p in params:
                    if isinstance(p, str) and p.strip():
                        param_clean = p.strip().lower().replace(" ", "_").replace("-", "_")
                        valid_params.append(param_clean)
                return valid_params
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        return []

    def _get_default_parameters(self, profile_type: str) -> List[str]:
        """Get fallback default parameters when memory/LLM is unavailable."""
        config = self.profile_config.get(profile_type, {})
        example_params = config.get("example_parameters", [])
        return example_params[:2] if len(example_params) >= 2 else example_params

