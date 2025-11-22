"""
InfoOrchestrator module that acts as an orchestrator agent for profile generation.

This module orchestrates the profile generation process:
- Determines required profiles (user/item) and generates schemas from memory
- Sends schemas to schema_fitter module for profile generation
- Consolidates results into final JSON for the reasoning module.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from websocietysimulator.agent.modules.parameter_retriever_module import ParameterRetriever

logger = logging.getLogger("websocietysimulator")


class InfoOrchestrator:
    """
    Orchestrator agent that coordinates profile generation for user and book/item profiles.
    """
    
    USER_PROFILE = "user"
    RESTAURANT_PROFILE = "restaurant"
    
    PROFILE_CONFIG = {
        USER_PROFILE: {
            "detection_keywords": [
                "user", "user's", "user profile", "user behavior", 
                "review history", "preference", "sentiment", "rating"
            ],
            "memory_query_keywords": ["user", "review", "preference", "sentiment"],
            "example_parameters": [
                "genre_preference", "reading_style", "theme_preference", 
                "author_preference", "review_sentiment"
            ]
        },
        RESTAURANT_PROFILE: {
            "detection_keywords": [
                "candidate", "metadata", "location", "categories", "book"
            ],
            "memory_query_keywords": ["restaurant", "business", "item", "category", "book"],
            "example_parameters": [
                "genre", "reading_level", "theme", "author_style", "rating"
            ]
        }
    }

    def __init__(self, memory=None, llm=None, schema_fitter=None, interaction_tool=None):
        self.memory = memory
        self.llm = llm
        self.schema_fitter = schema_fitter
        self.interaction_tool = interaction_tool
        self.parameter_retriever = ParameterRetriever(memory, llm, self.PROFILE_CONFIG)

    def __call__(
        self, 
        planner_steps: List[Dict[str, Any]], 
        user_id: Optional[str] = None,
        item_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Orchestrate profile generation: analyze steps, call schema_fitter, and consolidate results."""
        logger.info("InfoOrchestrator: Starting profile generation")
        logger.info(f"  Planner steps: {len(planner_steps)}, User ID: {user_id}, Item ID: {item_id}")
        
        # Generate profiles
        user_profile = self._generate_profile(planner_steps, user_id, self.USER_PROFILE) if user_id else {}
        item_profile = self._generate_profile(planner_steps, item_id, self.RESTAURANT_PROFILE) if item_id else {}
        
        # Consolidate results
        consolidated = {}
        if user_profile:
            consolidated["user_profile"] = user_profile
        if item_profile:
            consolidated["item_profile"] = item_profile
        
        logger.info(f"  Consolidated profiles: {list(consolidated.keys())}")
        return consolidated

    def _generate_profile(
        self, 
        planner_steps: List[Dict[str, Any]], 
        entity_id: Optional[str],
        profile_type: str
    ) -> Dict[str, Any]:
        """Generate a single profile if needed."""
        if not self._requires_profile(planner_steps, profile_type):
            return {}
        if not self.schema_fitter or not entity_id:
            return {}
        
        params = self.parameter_retriever.retrieve_parameters(planner_steps, profile_type)
        logger.info(f"  {profile_type.capitalize()} profile parameters: {params}")
        return self._call_schema_fitter(params, entity_id, profile_type)

    def _requires_profile(self, steps: List[Dict[str, Any]], profile_type: str) -> bool:
        """Check if any planner step requires the specified profile type."""
        keywords = self.PROFILE_CONFIG[profile_type]["detection_keywords"]
        step_text = json.dumps(steps, default=str).lower()
        return any(keyword in step_text for keyword in keywords)

    def _call_schema_fitter(
        self, 
        params: List[str], 
        entity_id: Optional[str],
        profile_type: str
    ) -> Dict[str, Any]:
        """Call schema_fitter to generate profile for user or book/item."""
        if not self.schema_fitter or not entity_id:
            return {}
        
        schema = {param: "string" for param in params}
        schema["user_id" if profile_type == self.USER_PROFILE else "item_id"] = "string"
        
        try:
            result = self.schema_fitter.build_profile(
                schema=schema,
                entity_id=entity_id,
                profile_type=profile_type,
                max_reviews=50
            )
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.error(f"Error calling schema_fitter: {e}", exc_info=True)
            return {}
