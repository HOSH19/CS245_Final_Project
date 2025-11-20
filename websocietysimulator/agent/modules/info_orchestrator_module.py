"""
InfoRetriever module that acts as an orchestrator agent for profile generation.

This module orchestrates the profile generation process:
- Stage 1: Determines required profiles (user/restaurant) and sends schemas to profile_agent
- Stage 2: Receives profile outputs and consolidates them into a final JSON for the reasoning module.
"""

import json
import re
from typing import Any, Dict, List, Optional


class InfoOrchestrator:
    """
    Orchestrator agent that coordinates profile generation for user and restaurant profiles.
    
    Memory Retrieval Flow:
    ---------------------
    When retrieving parameters from memory, the following happens:
    
    1. Build query from planner steps (descriptions/reasoning containing relevant keywords)
    2. Call memory module: memory(query) -> calls retriveMemory(query)
    3. Memory module returns: task_trajectory string from metadata
       - This is the full execution trajectory of how a similar task was completed
       - Contains: steps taken, parameters used, successful completion path
       - Stored as: metadata['task_trajectory'] in the memory database
    4. Feed task_trajectory to LLM along with current planner steps
    5. LLM analyzes the past trajectory to extract the most relevant parameter names
    6. Return extracted parameters (e.g., ["spice_level", "affordability"])
    
    The task_trajectory is essentially a "memory" of how a similar problem was solved before,
    and the LLM uses this to determine which profile parameters were most effective.
    """
    
    # Profile type constants
    USER_PROFILE = "user"
    RESTAURANT_PROFILE = "restaurant"
    
    # Keyword mappings for profile detection
    PROFILE_KEYWORDS = {
        USER_PROFILE: [
            "user", "user's", "user profile", "user behavior", 
            "review history", "preference", "sentiment", "rating"
        ],
        RESTAURANT_PROFILE: [
            "restaurant", "business", "item", "venue", "establishment",
            "candidate", "metadata", "location", "categories"
        ]
    }
    
    # Memory query keywords
    MEMORY_QUERY_KEYWORDS = {
        USER_PROFILE: ["user", "review", "preference", "sentiment"],
        RESTAURANT_PROFILE: ["restaurant", "business", "item", "category"]
    }
    
    # Common parameters for each profile type
    COMMON_PARAMETERS = {
        USER_PROFILE: [
            "spice_level", "affordability", "cuisine_type", "dietary_restrictions",
            "price_range", "ambiance", "service_quality", "location_preference",
            "rating_preference", "review_sentiment", "category_preference"
        ],
        RESTAURANT_PROFILE: [
            "cuisine_type", "price_range", "ambiance", "service_quality",
            "location", "rating", "popularity", "category", "atmosphere",
            "price_level", "cuisine_style"
        ]
    }
    
    # Default parameter mappings based on keywords
    DEFAULT_PARAMETER_MAPPINGS = {
        USER_PROFILE: {
            ("spice", "spicy"): "spice_level",
            ("price", "afford", "cost"): "affordability",
            ("cuisine", "food"): "cuisine_type",
            ("category", "type"): "category_preference",
            ("sentiment", "rating"): "review_sentiment"
        },
        RESTAURANT_PROFILE: {
            ("cuisine", "food"): "cuisine_type",
            ("price", "afford"): "price_range",
            ("location", "city"): "location",
            ("rating", "star"): "rating"
        }
    }
    
    # Fallback defaults
    FALLBACK_DEFAULTS = {
        USER_PROFILE: ["affordability", "category_preference"],
        RESTAURANT_PROFILE: ["cuisine_type", "price_range"]
    }

    def __init__(self, memory=None, llm=None, profile_agent=None, interaction_tool=None):
        """
        Initialize the InfoRetriever orchestrator module.
        
        Args:
            memory: Memory module instance for retrieving relevant parameters
            llm: Optional LLM instance for advanced analysis
            profile_agent: Profile agent module that generates profiles
            interaction_tool: Tool for fetching user/item data
        """
        self.memory = memory
        self.llm = llm
        self.profile_agent = profile_agent
        self.interaction_tool = interaction_tool

    def __call__(
        self, 
        planner_steps: List[Dict[str, Any]], 
        user_id: Optional[str] = None,
        item_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate profile generation: analyze steps, call profile_agent, and consolidate results.
        
        Args:
            planner_steps: List of step dictionaries from the Planner module.
                          Each step typically contains:
                          - "step": step number
                          - "description": subtask description
                          - "reasoning": reasoning instruction
            user_id: Optional user ID for fetching user data
            item_id: Optional item/business ID for fetching restaurant data
        
        Returns:
            Consolidated JSON dictionary containing both user and restaurant profiles.
            Structure: {
                "user_profile": {...},
                "restaurant_profile": {...}
            }
        """
        # Stage 1: Determine what profiles are needed and get parameter schemas
        needs_user = self._requires_profile(planner_steps, self.USER_PROFILE)
        needs_restaurant = self._requires_profile(planner_steps, self.RESTAURANT_PROFILE)
        
        # Stage 1: Get parameter schemas from memory and call profile_agent
        user_profile_result = {}
        if needs_user:
            user_params = self._retrieve_parameters_from_memory(planner_steps, self.USER_PROFILE)
            if self.profile_agent:
                user_profile_result = self._call_profile_agent(
                    user_params, user_id, self.USER_PROFILE
                )
        
        restaurant_profile_result = {}
        if needs_restaurant:
            restaurant_params = self._retrieve_parameters_from_memory(planner_steps, self.RESTAURANT_PROFILE)
            if self.profile_agent:
                restaurant_profile_result = self._call_profile_agent(
                    restaurant_params, item_id, self.RESTAURANT_PROFILE
                )
        
        # Stage 2: Consolidate results into final JSON
        return self._consolidate_profiles(user_profile_result, restaurant_profile_result)

    def _requires_profile(self, steps: List[Dict[str, Any]], profile_type: str) -> bool:
        """Check if any planner step requires the specified profile type."""
        keywords = self.PROFILE_KEYWORDS[profile_type]
        step_text = json.dumps(steps, default=str).lower()
        return any(keyword in step_text for keyword in keywords)

    def _retrieve_parameters_from_memory(
        self, 
        steps: List[Dict[str, Any]], 
        profile_type: str
    ) -> List[str]:
        """
        Retrieve relevant parameter names from memory using LLM to extract best parameters.
        
        The memory module returns a 'task_trajectory' string which contains:
        - The full execution trajectory of how a similar task was completed in the past
        - Information about what steps were taken, what parameters were used, and how the task succeeded
        - This trajectory is stored in memory from previous successful task completions
        
        Args:
            steps: List of planner step dictionaries
            profile_type: "user" or "restaurant"
            
        Returns:
            List of parameter names (e.g., ["spice_level", "affordability"])
        """
        # Early returns for missing dependencies
        if not self.llm or not self.memory:
            return self._get_default_parameters(steps, profile_type)
        
        # Query memory for relevant information
        # The query is built from planner step descriptions/reasoning
        query = self._build_memory_query(steps, profile_type)
        
        # Call memory module: memory(query) -> retriveMemory(query) -> returns task_trajectory string
        # The task_trajectory contains the full execution path of a similar past task
        memory_result = self.memory(query) if query else ""
        
        if not memory_result:
            return self._get_default_parameters(steps, profile_type)
        
        # Feed memory result (task_trajectory string) to LLM to extract best parameters
        # The LLM analyzes the past trajectory to determine which parameters were most useful
        parameters = self._extract_best_parameters_with_llm(memory_result, steps, profile_type)
        return parameters or self._get_default_parameters(steps, profile_type)

    def _build_memory_query(self, steps: List[Dict[str, Any]], profile_type: str) -> str:
        """Build a query string from planner steps to search memory."""
        keywords = self.MEMORY_QUERY_KEYWORDS[profile_type]
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
        """
        Use LLM to extract the best parameter names from memory content.
        
        Args:
            memory_result: The task_trajectory string retrieved from memory. This contains:
                - The full execution trajectory of a similar past task
                - Information about successful parameter usage in previous tasks
                - The complete path taken to solve a similar problem
                - Stored as metadata['task_trajectory'] in the memory module
            steps: Planner steps for context
            profile_type: "user" or "restaurant"
            
        Returns:
            List of parameter names extracted by LLM
            
        Note:
            The memory_result is the 'task_trajectory' field from memory metadata, which is a string
            containing the complete execution history of how a similar task was successfully completed.
            The LLM analyzes this trajectory to identify which profile parameters were most relevant
            and effective in the past, then extracts them for use in the current task.
        """
        if not self.llm or not memory_result:
            return []
        
        common_parameters = self.COMMON_PARAMETERS[profile_type]
        profile_description = "user profile" if profile_type == self.USER_PROFILE else "restaurant/business profile"
        step_context = json.dumps(steps, default=str, indent=2)
        
        prompt = f"""You are analyzing planner steps and memory content to determine the best parameters 
for a {profile_description} schema.

Your task: Based on the memory content and planner steps, identify the most relevant and important 
parameters that should be included in the {profile_type} profile schema. These parameters will be used 
to structure profile data for recommendation systems.

Planner Steps:
{step_context}

Memory Content (task_trajectory from similar past experiences):
This is the full execution trajectory of how a similar task was successfully completed in the past.
It contains information about what steps were taken, what parameters were used, and how the task succeeded.
Analyze this trajectory to identify which profile parameters were most effective.

{memory_result}

Common parameters you might consider (but you can suggest others if more relevant):
{', '.join(common_parameters)}

Analyze the memory content carefully. What parameters are most relevant based on:
1. What the planner steps are trying to accomplish
2. What information is available in the memory
3. What would be most useful for building an effective profile

Return ONLY a JSON array of parameter names (as strings). Be selective - choose the 3-5 most important 
and relevant parameters. Use snake_case naming.

Example output format: ["spice_level", "affordability", "cuisine_type"]

Return only the JSON array, no additional text or explanation:
"""
        
        try:
            response = self.llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return self._parse_llm_response(response)
        except Exception:
            return []

    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse and validate LLM response to extract parameter list."""
        response_clean = response.strip()
        
        # Remove markdown code blocks if present
        if "```" in response_clean:
            json_match = re.search(r'\[.*?\]', response_clean, re.DOTALL)
            if json_match:
                response_clean = json_match.group(0)
        
        try:
            params = json.loads(response_clean)
            if isinstance(params, list):
                # Validate and clean parameter names
                valid_params = []
                for p in params:
                    if isinstance(p, str) and p.strip():
                        # Ensure snake_case format
                        param_clean = p.strip().lower().replace(" ", "_").replace("-", "_")
                        valid_params.append(param_clean)
                return valid_params
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        return []

    def _get_default_parameters(self, steps: List[Dict[str, Any]], profile_type: str) -> List[str]:
        """Get default parameters based on step analysis."""
        step_text = json.dumps(steps, default=str).lower()
        defaults = []
        mappings = self.DEFAULT_PARAMETER_MAPPINGS[profile_type]
        
        for keywords, param_name in mappings.items():
            if any(keyword in step_text for keyword in keywords):
                defaults.append(param_name)
        
        return defaults or self.FALLBACK_DEFAULTS[profile_type]

    def _call_profile_agent(
        self, 
        params: List[str], 
        entity_id: Optional[str],
        profile_type: str
    ) -> Dict[str, Any]:
        """
        Call profile_agent to generate profile for user or restaurant.
        
        Args:
            params: List of parameter names to include in schema
            entity_id: User ID or item ID for fetching data
            profile_type: "user" or "restaurant"
            
        Returns:
            Dictionary with profile result from profile_agent
        """
        if not self.profile_agent or not self.interaction_tool or not entity_id:
            return {}
        
        # Fetch data based on profile type
        if profile_type == self.USER_PROFILE:
            profile_data = self.interaction_tool.get_user(user_id=entity_id)
            reviews_data = self.interaction_tool.get_reviews(user_id=entity_id)
        else:
            profile_data = self.interaction_tool.get_item(item_id=entity_id)
            reviews_data = self.interaction_tool.get_reviews(item_id=entity_id)
        
        # Call profile_agent
        try:
            result = self.profile_agent(
                user_profile=profile_data,
                user_reviews=reviews_data,
                interaction_tool=self.interaction_tool
            )
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}

    def _consolidate_profiles(
        self, 
        user_profile_result: Dict[str, Any],
        restaurant_profile_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Consolidate user and restaurant profiles into final JSON structure.
        
        Args:
            user_profile_result: User profile output from profile_agent
            restaurant_profile_result: Restaurant profile output from profile_agent
            
        Returns:
            Consolidated dictionary with both profiles
        """
        consolidated = {}
        if user_profile_result:
            consolidated["user_profile"] = user_profile_result
        if restaurant_profile_result:
            consolidated["restaurant_profile"] = restaurant_profile_result
        return consolidated
