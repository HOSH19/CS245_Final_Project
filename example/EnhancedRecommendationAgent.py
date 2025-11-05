"""
Enhanced Recommendation Agent using Planning, Reasoning, and Memory Modules

This agent demonstrates proper usage of the modular architecture available in the
websocietysimulator framework for Track 2 (Recommendation).
"""

import json
import re
import logging
import tiktoken
from typing import List
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import (
    PlanningVoyager, PlanningIO, PlanningOPENAGI, PlanningHUGGINGGPT, PlanningTD, PlanningDEPS
)
from websocietysimulator.agent.modules.reasoning_modules import (
    ReasoningCOT, ReasoningStepBack, ReasoningSelfRefine, ReasoningCOTSC, ReasoningTOT, ReasoningIO
)
from websocietysimulator.agent.modules.memory_modules import (
    MemoryGenerative, MemoryDILU, MemoryVoyager, MemoryTP
)

logging.basicConfig(level=logging.INFO)


def num_tokens_from_string(string: str) -> int:
    """Count the number of tokens in a string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        return len(encoding.encode(string))
    except Exception as e:
        logging.warning(f"Error encoding string: {e}")
        return 0


def truncate_to_token_limit(string: str, max_tokens: int = 12000) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        tokens = encoding.encode(string)
        if len(tokens) <= max_tokens:
            return string
        return encoding.decode(tokens[:max_tokens])
    except Exception as e:
        logging.warning(f"Error truncating string: {e}")
        return string[:max_tokens * 3]  # Rough approximation


class EnhancedRecommendationPlanning(PlanningIO):
    """
    Custom planning module for recommendation tasks.
    Creates a structured plan to gather user preferences and item information.
    """
    
    def __init__(self, llm):
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """
        Create a planning prompt specifically for recommendation tasks.
        """
        if feedback == '':
            prompt = '''You are a planning agent for a recommendation task.
Your goal is to create a systematic plan to understand user preferences and rank items accordingly.

Here is an example plan:
sub-task 1: {"description": "Retrieve and analyze user profile and preferences", "reasoning instruction": "Understand what the user likes and dislikes based on their profile"}
sub-task 2: {"description": "Retrieve and analyze user's review history", "reasoning instruction": "Extract patterns in user preferences, rating behavior, and interests"}
sub-task 3: {"description": "Retrieve detailed information for all candidate items", "reasoning instruction": "Understand each item's characteristics, ratings, and features"}
sub-task 4: {"description": "Match user preferences with item characteristics", "reasoning instruction": "Rank items by how well they match the user's demonstrated preferences"}

Now create a plan for this recommendation task:
Task Type: {task_type}
Task Description: {task_description}

Output your plan as a series of sub-tasks in the same format.
'''
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = f'''You are a planning agent for a recommendation task.
Based on the following feedback, adjust your planning strategy:

Feedback: {feedback}

Task Type: {task_type}
Task Description: {task_description}

Create an improved plan that addresses the feedback.
'''
        return prompt


class EnhancedRecommendationReasoning(ReasoningStepBack):
    """
    Custom reasoning module using Step-Back prompting for better recommendations.
    Step-back prompting helps the model think about high-level principles first.
    """
    
    def __init__(self, profile_type_prompt, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
    
    def __call__(self, task_description: str, feedback: str = ''):
        """
        Generate recommendations using step-back reasoning.
        First, extract general principles about the user's preferences,
        then apply them to rank items.
        """
        # First, step back to understand general principles
        principles = self.stepback(task_description)
        
        # Then apply those principles to the specific task
        prompt = f'''You are a recommendation system. 

High-level understanding of user preferences:
{principles}

Now, based on this understanding, solve the following task:

{task_description}

Think step-by-step:
1. What are the key factors this user values?
2. Which items best match these factors?
3. Rank the items from most to least suitable

Your output must be ONLY a ranked list in this format:
['item_id_1', 'item_id_2', 'item_id_3', ...]

Do not include any other text or explanation in your final answer.
'''
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=2000
        )
        
        return reasoning_result
    
    def stepback(self, task_description):
        """
        Step back to understand high-level principles about user preferences.
        """
        stepback_prompt = f'''Before making recommendations, let's understand the general patterns:

Given the user's profile and review history, what are the key principles that guide this user's preferences?

Consider:
- What types of items does this user typically enjoy?
- What characteristics or features are most important to them?
- What patterns exist in their rating behavior?
- What categories or genres do they prefer?

Task information:
{task_description[:2000]}

Provide a concise summary of the user's preference principles (2-3 sentences).
'''
        
        messages = [{"role": "user", "content": stepback_prompt}]
        principle = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )
        return principle


class EnhancedRecommendationAgent(RecommendationAgent):
    """
    Enhanced Recommendation Agent that demonstrates proper usage of
    Planning, Reasoning, and Memory modules.
    
    This agent:
    1. Uses a planning module to structure the recommendation process
    2. Uses step-back reasoning for better understanding of user preferences
    3. Uses memory to store and retrieve patterns from user reviews
    """
    
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        
        # Initialize planning module
        self.planning = EnhancedRecommendationPlanning(llm=self.llm)
        
        # Initialize reasoning module with StepBack
        self.reasoning = EnhancedRecommendationReasoning(
            profile_type_prompt='You are an intelligent recommendation system.',
            llm=self.llm
        )
        
        # Initialize memory module for storing review patterns
        self.memory = MemoryGenerative(llm=self.llm)
        
        logging.info("EnhancedRecommendationAgent initialized with Planning, Reasoning, and Memory modules")
    
    def workflow(self) -> List[str]:
        """
        Main workflow for generating recommendations.
        
        Returns:
            list: A ranked list of item IDs, from most to least recommended
        """
        try:
            # Step 1: Create a plan (using a simple predefined plan for efficiency)
            logging.info(f"Creating recommendation plan for user {self.task['user_id']}")
            
            plan = [
                {
                    'description': 'Retrieve and analyze user profile',
                    'reasoning instruction': 'Understand user preferences'
                },
                {
                    'description': 'Retrieve and analyze user review history',
                    'reasoning instruction': 'Extract preference patterns'
                },
                {
                    'description': 'Retrieve candidate item information',
                    'reasoning instruction': 'Understand item characteristics'
                }
            ]
            
            # Step 2: Execute the plan - gather information
            user_info = None
            user_reviews = []
            candidate_items = []
            
            for sub_task in plan:
                description = sub_task['description'].lower()
                
                if 'user profile' in description:
                    # Get user information
                    user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
                    user_summary = self._create_user_summary(user_info)
                    logging.info(f"Retrieved user profile for {self.task['user_id']}")
                
                elif 'review history' in description or 'user review' in description:
                    # Get user's review history
                    user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
                    
                    # Store important reviews in memory
                    if user_reviews:
                        for review in user_reviews[:30]:  # Store up to 30 reviews
                            if 'text' in review and review['text']:
                                review_summary = f"Stars: {review.get('stars', 'N/A')}, Text: {review['text'][:200]}"
                                self.memory(f"review: {review_summary}")
                    
                    logging.info(f"Retrieved {len(user_reviews)} reviews from user history")
                
                elif 'candidate item' in description or 'item information' in description:
                    # Get information for all candidate items
                    for item_id in self.task['candidate_list']:
                        try:
                            item = self.interaction_tool.get_item(item_id=item_id)
                            if item:
                                # Extract only relevant fields to save tokens
                                filtered_item = self._filter_item_info(item)
                                candidate_items.append(filtered_item)
                        except Exception as e:
                            logging.warning(f"Error retrieving item {item_id}: {e}")
                            # Add minimal info as fallback
                            candidate_items.append({'item_id': item_id})
                    
                    logging.info(f"Retrieved information for {len(candidate_items)} candidate items")
            
            # Step 3: Analyze user preferences using memory
            user_preference_summary = self._analyze_user_preferences(user_reviews)
            
            # Step 4: Create comprehensive task description for reasoning
            task_description = self._create_recommendation_prompt(
                user_info=user_info,
                user_reviews=user_reviews,
                candidate_items=candidate_items,
                user_preference_summary=user_preference_summary
            )
            
            # Step 5: Use reasoning module to generate recommendations
            logging.info("Generating recommendations using reasoning module")
            result = self.reasoning(task_description)
            
            # Step 6: Parse the result to extract the ranked list
            ranked_list = self._parse_recommendation_result(result)
            
            # Step 7: Validate and return the list
            validated_list = self._validate_recommendations(ranked_list, self.task['candidate_list'])
            
            logging.info(f"Generated {len(validated_list)} recommendations")
            return validated_list
            
        except Exception as e:
            logging.error(f"Error in workflow: {e}", exc_info=True)
            # Return the candidate list as-is if there's an error
            return self.task['candidate_list']
    
    def _create_user_summary(self, user_info):
        """Create a concise summary of user information."""
        if not user_info:
            return "User information not available."
        
        summary = {}
        
        # Extract key fields
        if 'user_id' in user_info:
            summary['user_id'] = user_info['user_id']
        if 'name' in user_info:
            summary['name'] = user_info['name']
        if 'average_stars' in user_info or 'stars' in user_info:
            summary['avg_rating'] = user_info.get('average_stars') or user_info.get('stars')
        if 'review_count' in user_info:
            summary['review_count'] = user_info['review_count']
        
        return summary
    
    def _filter_item_info(self, item):
        """Filter item information to keep only relevant fields."""
        # Different platforms have different field names
        keys_to_extract = [
            'item_id', 'business_id', 'asin',  # ID fields
            'name', 'title', 'title_without_series',  # Name fields
            'stars', 'average_rating', 'ratings_count',  # Rating fields
            'review_count', 'rating_number',  # Review count fields
            'categories', 'attributes',  # Category/attribute fields
            'description',  # Description
            'price', 'authors', 'publisher'  # Additional useful fields
        ]
        
        filtered = {}
        for key in keys_to_extract:
            if key in item and item[key] is not None:
                value = item[key]
                # Truncate long strings
                if isinstance(value, str) and len(value) > 300:
                    filtered[key] = value[:300] + "..."
                elif isinstance(value, dict):
                    # For nested dicts, convert to string and truncate
                    filtered[key] = str(value)[:200]
                else:
                    filtered[key] = value
        
        return filtered
    
    def _analyze_user_preferences(self, user_reviews):
        """
        Analyze user reviews to extract preference patterns.
        """
        if not user_reviews or len(user_reviews) == 0:
            return "No review history available for this user."
        
        # Calculate statistics
        total_reviews = len(user_reviews)
        ratings = [r.get('stars', 0) for r in user_reviews if 'stars' in r]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Count rating distribution
        rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for rating in ratings:
            rounded = round(rating)
            if rounded in rating_dist:
                rating_dist[rounded] += 1
        
        # Extract sample reviews from different rating levels
        high_rated = [r for r in user_reviews if r.get('stars', 0) >= 4.0]
        low_rated = [r for r in user_reviews if r.get('stars', 0) <= 2.0]
        
        summary = f"""User Review Analysis:
- Total Reviews: {total_reviews}
- Average Rating Given: {avg_rating:.2f}
- Rating Distribution: {rating_dist}
- High-rated items: {len(high_rated)} ({len(high_rated)/total_reviews*100:.1f}%)
- Low-rated items: {len(low_rated)} ({len(low_rated)/total_reviews*100:.1f}%)

Sample of highly-rated reviews (what user likes):
"""
        
        for i, review in enumerate(high_rated[:3]):
            text = review.get('text', '')[:150]
            summary += f"\n{i+1}. ({review.get('stars', 'N/A')} stars) {text}..."
        
        if low_rated:
            summary += "\n\nSample of low-rated reviews (what user dislikes):\n"
            for i, review in enumerate(low_rated[:2]):
                text = review.get('text', '')[:150]
                summary += f"\n{i+1}. ({review.get('stars', 'N/A')} stars) {text}..."
        
        return summary
    
    def _create_recommendation_prompt(self, user_info, user_reviews, candidate_items, user_preference_summary):
        """
        Create a comprehensive prompt for the reasoning module.
        """
        # Prepare user review history (truncate if too long)
        review_history_str = str(user_reviews)
        if num_tokens_from_string(review_history_str) > 8000:
            review_history_str = truncate_to_token_limit(review_history_str, 8000)
        
        # Prepare candidate items (truncate if too long)
        candidate_items_str = str(candidate_items)
        if num_tokens_from_string(candidate_items_str) > 6000:
            candidate_items_str = truncate_to_token_limit(candidate_items_str, 6000)
        
        prompt = f"""You are a personalized recommendation system. Your task is to rank items based on how well they match a specific user's preferences.

USER PROFILE:
{user_info}

USER PREFERENCE ANALYSIS:
{user_preference_summary}

CANDIDATE ITEMS TO RANK:
Item IDs: {self.task['candidate_list']}

DETAILED ITEM INFORMATION:
{candidate_items_str}

INSTRUCTIONS:
1. Analyze the user's preferences based on their review history
2. Evaluate each candidate item against these preferences
3. Rank items from MOST suitable to LEAST suitable for this user
4. Consider:
   - Items similar to what the user rated highly in the past
   - Categories/genres the user prefers
   - Rating patterns and quality expectations
   - Specific features or attributes the user values

CRITICAL OUTPUT REQUIREMENTS:
- Your output MUST be ONLY a Python list of item IDs
- Use the EXACT item IDs from the candidate list: {self.task['candidate_list']}
- Do NOT introduce any new IDs
- Do NOT include explanations or other text
- Format: ['item_id_1', 'item_id_2', 'item_id_3', ...]

Example of correct format:
['B001', 'B002', 'B003']

Now rank the items:
"""
        
        return prompt
    
    def _parse_recommendation_result(self, result):
        """
        Parse the LLM result to extract the ranked list of item IDs.
        """
        try:
            # Look for a list pattern in the result
            # Try to find content between square brackets
            match = re.search(r'\[.*?\]', result, re.DOTALL)
            
            if match:
                list_str = match.group()
                # Use eval to parse the list (be careful with this in production!)
                try:
                    ranked_list = eval(list_str)
                    if isinstance(ranked_list, list):
                        logging.info(f"Successfully parsed recommendation list with {len(ranked_list)} items")
                        return ranked_list
                except:
                    logging.warning("Failed to eval the matched list string")
            
            # If the above didn't work, try to extract quoted strings
            items = re.findall(r'["\']([^"\']+)["\']', result)
            if items:
                logging.info(f"Extracted {len(items)} items using regex")
                return items
            
            logging.warning("Could not parse recommendation result")
            return []
            
        except Exception as e:
            logging.error(f"Error parsing recommendation result: {e}")
            return []
    
    def _validate_recommendations(self, ranked_list, candidate_list):
        """
        Validate and fix the recommendation list.
        Ensures all items are from the candidate list and no duplicates exist.
        """
        # Remove duplicates while preserving order
        seen = set()
        unique_list = []
        for item_id in ranked_list:
            if item_id not in seen and item_id in candidate_list:
                seen.add(item_id)
                unique_list.append(item_id)
        
        # Add any missing items from candidate list
        for item_id in candidate_list:
            if item_id not in seen:
                unique_list.append(item_id)
        
        logging.info(f"Validated list: {len(unique_list)} items (original: {len(ranked_list)})")
        return unique_list
    
    # ========================================================================
    # REFACTORED HELPER METHODS TO REDUCE DUPLICATION
    # ========================================================================
    
    def _gather_user_data(self):
        """
        Gather user profile and review history.
        
        Returns:
            tuple: (user_info, user_reviews)
        """
        user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
        user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
        
        logging.info(f"Retrieved user profile and {len(user_reviews)} reviews")
        return user_info, user_reviews
    
    def _gather_candidate_items(self):
        """
        Gather information for all candidate items.
        
        Returns:
            list: List of filtered item information dictionaries
        """
        candidate_items = []
        for item_id in self.task['candidate_list']:
            try:
                item = self.interaction_tool.get_item(item_id=item_id)
                if item:
                    candidate_items.append(self._filter_item_info(item))
            except Exception as e:
                logging.warning(f"Error retrieving item {item_id}: {e}")
                candidate_items.append({'item_id': item_id})
        
        logging.info(f"Retrieved information for {len(candidate_items)} candidate items")
        return candidate_items
    
    def _store_reviews_in_memory(self, user_reviews, memory_module, max_reviews=20):
        """
        Store user reviews in a given memory module.
        
        Args:
            user_reviews: List of user review dictionaries
            memory_module: Memory module instance to store reviews in
            max_reviews: Maximum number of reviews to store
        """
        if user_reviews and memory_module:
            for review in user_reviews[:max_reviews]:
                if 'text' in review and review['text']:
                    review_summary = f"Stars: {review.get('stars', 'N/A')}, Text: {review['text'][:200]}"
                    memory_module(f"review: {review_summary}")
            
            logging.info(f"Stored {min(len(user_reviews), max_reviews)} reviews in memory")
    
    def _generate_recommendations_with_reasoning(self, reasoning_module, user_info, 
                                                  user_reviews, candidate_items, 
                                                  additional_context=""):
        """
        Generate recommendations using a given reasoning module.
        
        Args:
            reasoning_module: Reasoning module instance to use
            user_info: User profile information
            user_reviews: List of user reviews
            candidate_items: List of candidate item information
            additional_context: Optional additional context to include in prompt
        
        Returns:
            list: Validated ranked list of item IDs
        """
        # Create preference summary
        user_preference_summary = self._analyze_user_preferences(user_reviews)
        
        # Add any additional context
        if additional_context:
            user_preference_summary += f"\n\n{additional_context}"
        
        # Create task description
        task_description = self._create_recommendation_prompt(
            user_info=user_info,
            user_reviews=user_reviews,
            candidate_items=candidate_items,
            user_preference_summary=user_preference_summary
        )
        
        # Generate recommendations using reasoning module
        result = reasoning_module(task_description)
        
        # Parse and validate
        ranked_list = self._parse_recommendation_result(result)
        validated_list = self._validate_recommendations(ranked_list, self.task['candidate_list'])
        
        return validated_list
    
    def _execute_generic_workflow(self, workflow_name, planning_module=None, 
                                   reasoning_module=None, memory_module=None):
        """
        Generic workflow template that can be customized with different modules.
        
        Args:
            workflow_name: Name of the workflow (for logging)
            planning_module: Optional planning module to use (if None, uses simple plan)
            reasoning_module: Reasoning module to use (if None, uses self.reasoning)
            memory_module: Optional memory module to use
        
        Returns:
            list: A ranked list of item IDs
        """
        try:
            logging.info(f"Executing {workflow_name}")
            
            # Step 1: Create plan (if planning module provided)
            if planning_module:
                plan_task = f"Create recommendations for user {self.task['user_id']} from {len(self.task['candidate_list'])} items"
                plan = planning_module(
                    task_type='Recommendation',
                    task_description=plan_task,
                    feedback='',
                    few_shot='sub-task 1: {"description": "Get user preferences", "reasoning instruction": "Analyze user history"}'
                )
                logging.info(f"Generated plan with {len(plan)} subtasks")
            
            # Step 2: Gather user data
            user_info, user_reviews = self._gather_user_data()
            
            # Step 3: Store reviews in memory (if memory module provided)
            if memory_module:
                self._store_reviews_in_memory(user_reviews, memory_module)
                
                # Retrieve memory context
                additional_context = ""
                if user_reviews and len(user_reviews) > 0:
                    sample_review = user_reviews[0].get('text', '')
                    if sample_review:
                        memory_context = memory_module(sample_review[:200])
                        additional_context = f"Memory Context:\n{memory_context}"
            else:
                additional_context = ""
            
            # Step 4: Gather candidate items
            candidate_items = self._gather_candidate_items()
            
            # Step 5: Generate recommendations
            if reasoning_module is None:
                reasoning_module = self.reasoning
            
            validated_list = self._generate_recommendations_with_reasoning(
                reasoning_module=reasoning_module,
                user_info=user_info,
                user_reviews=user_reviews,
                candidate_items=candidate_items,
                additional_context=additional_context
            )
            
            logging.info(f"{workflow_name} generated {len(validated_list)} recommendations")
            return validated_list
            
        except Exception as e:
            logging.error(f"Error in {workflow_name}: {e}", exc_info=True)
            return self.task['candidate_list']
    
    # ========================================================================
    # ALTERNATIVE WORKFLOW METHODS USING DIFFERENT MODULE COMBINATIONS
    # ========================================================================
    
    def workflow_with_voyager_planning(self) -> List[str]:
        """
        Alternative workflow using PlanningVoyager module.
        
        PlanningVoyager creates subgoals-based plans with detailed decomposition.
        Good for complex recommendation tasks requiring multiple steps.
        
        Returns:
            list: A ranked list of item IDs
        """
        voyager_planning = PlanningVoyager(llm=self.llm)
        return self._execute_generic_workflow(
            workflow_name="Voyager Planning",
            planning_module=voyager_planning,
            reasoning_module=self.reasoning,
            memory_module=None
        )
    
    def workflow_with_self_refine(self) -> List[str]:
        """
        Alternative workflow using ReasoningSelfRefine module.
        
        Self-refine generates an initial recommendation, then reflects and refines it.
        Good for improving recommendation quality through iteration.
        
        Returns:
            list: A ranked list of item IDs
        """
        self_refine_reasoning = ReasoningSelfRefine(
            profile_type_prompt='You are an intelligent recommendation system.',
            memory=None,
            llm=self.llm
        )
        return self._execute_generic_workflow(
            workflow_name="Self-Refine",
            planning_module=None,
            reasoning_module=self_refine_reasoning,
            memory_module=None
        )
    
    def workflow_with_cot_sc(self) -> List[str]:
        """
        Alternative workflow using ReasoningCOTSC (Chain-of-Thought with Self-Consistency).
        
        Generates multiple reasoning paths and selects the most consistent answer.
        Good for improving reliability through consensus.
        Note: This uses more API calls (n=5) so it's more expensive.
        
        Returns:
            list: A ranked list of item IDs
        """
        cot_sc_reasoning = ReasoningCOTSC(
            profile_type_prompt='You are an intelligent recommendation system.',
            memory=None,
            llm=self.llm
        )
        return self._execute_generic_workflow(
            workflow_name="COT-SC (Self-Consistency)",
            planning_module=None,
            reasoning_module=cot_sc_reasoning,
            memory_module=None
        )
    
    def workflow_with_voyager_memory(self) -> List[str]:
        """
        Alternative workflow using MemoryVoyager module.
        
        MemoryVoyager summarizes trajectories before storing them, providing
        concise memory retrieval. Good for learning from past recommendations.
        
        Returns:
            list: A ranked list of item IDs
        """
        voyager_memory = MemoryVoyager(llm=self.llm)
        return self._execute_generic_workflow(
            workflow_name="Voyager Memory",
            planning_module=None,
            reasoning_module=self.reasoning,
            memory_module=voyager_memory
        )
    
    def workflow_with_openagi_planning(self) -> List[str]:
        """
        Alternative workflow using PlanningOPENAGI module.
        
        OpenAGI planning creates concise todo lists with minimal, relevant tasks.
        Good for efficient, streamlined recommendation generation.
        
        Returns:
            list: A ranked list of item IDs
        """
        openagi_planning = PlanningOPENAGI(llm=self.llm)
        return self._execute_generic_workflow(
            workflow_name="OpenAGI Planning",
            planning_module=openagi_planning,
            reasoning_module=self.reasoning,
            memory_module=None
        )
    
    def workflow_hybrid_advanced(self) -> List[str]:
        """
        Advanced hybrid workflow combining multiple sophisticated modules:
        - PlanningHUGGINGGPT for dependency-aware planning
        - MemoryTP for trajectory-based planning from past experiences  
        - ReasoningCOT for step-by-step recommendation generation
        
        This is the most sophisticated approach, combining the best of all modules.
        Good for maximizing recommendation quality when API costs aren't a concern.
        
        Returns:
            list: A ranked list of item IDs
        """
        # Initialize all modules
        huggingpt_planning = PlanningHUGGINGGPT(llm=self.llm)
        tp_memory = MemoryTP(llm=self.llm)
        cot_reasoning = ReasoningCOT(
            profile_type_prompt='You are an intelligent recommendation system.',
            memory=tp_memory,
            llm=self.llm
        )
        
        # Use generic workflow with all three advanced modules
        return self._execute_generic_workflow(
            workflow_name="Hybrid Advanced (HuggingGPT + TP Memory + COT)",
            planning_module=huggingpt_planning,
            reasoning_module=cot_reasoning,
            memory_module=tp_memory
        )
    
    # ========================================================================
    # NEW WORKFLOW COMBINATIONS - EXPLORING UNUSED MODULES
    # ========================================================================
    
    def workflow_with_tot_reasoning(self) -> List[str]:
        """
        Workflow using ReasoningTOT (Tree of Thoughts).
        
        Generates 3 reasoning paths, then uses 5 LLM votes to select the best one.
        Most sophisticated reasoning approach - explores multiple strategies.
        
        ⚠️ WARNING: Very expensive (8 API calls per task: 3 reasoning + 5 voting)
        
        Returns:
            list: A ranked list of item IDs
        """
        tot_reasoning = ReasoningTOT(
            profile_type_prompt='You are an intelligent recommendation system.',
            memory=None,
            llm=self.llm
        )
        return self._execute_generic_workflow(
            workflow_name="Tree of Thoughts (TOT)",
            planning_module=None,
            reasoning_module=tot_reasoning,
            memory_module=None
        )
    
    def workflow_with_td_planning(self) -> List[str]:
        """
        Workflow using PlanningTD (Temporal Dependencies).
        
        Creates plans with explicit temporal dependencies and ordering.
        Good when task sequence matters (e.g., must get user data before items).
        
        Returns:
            list: A ranked list of item IDs
        """
        td_planning = PlanningTD(llm=self.llm)
        return self._execute_generic_workflow(
            workflow_name="Temporal Dependencies Planning",
            planning_module=td_planning,
            reasoning_module=self.reasoning,
            memory_module=None
        )
    
    def workflow_with_deps_planning(self) -> List[str]:
        """
        Workflow using PlanningDEPS (Multi-Hop Decomposition).
        
        Designed specifically for multi-hop reasoning tasks.
        Perfect for recommendations: user → reviews → items → ranking.
        Should perform very well due to alignment with task structure.
        
        Returns:
            list: A ranked list of item IDs
        """
        deps_planning = PlanningDEPS(llm=self.llm)
        generative_memory = MemoryGenerative(llm=self.llm)
        return self._execute_generic_workflow(
            workflow_name="Multi-Hop DEPS Planning",
            planning_module=deps_planning,
            reasoning_module=ReasoningCOT(
                profile_type_prompt='You are an intelligent recommendation system.',
                memory=generative_memory,
                llm=self.llm
            ),
            memory_module=generative_memory
        )
    
    def workflow_all_voyager(self) -> List[str]:
        """
        Full Voyager stack - all Voyager-style modules.
        
        Uses consistent subgoal-based approach across planning, reasoning, and memory.
        Provides cohesive framework with all modules working in same paradigm.
        
        Returns:
            list: A ranked list of item IDs
        """
        voyager_planning = PlanningVoyager(llm=self.llm)
        voyager_memory = MemoryVoyager(llm=self.llm)
        cot_reasoning = ReasoningCOT(
            profile_type_prompt='You are an intelligent recommendation system.',
            memory=voyager_memory,
            llm=self.llm
        )
        return self._execute_generic_workflow(
            workflow_name="All Voyager Stack",
            planning_module=voyager_planning,
            reasoning_module=cot_reasoning,
            memory_module=voyager_memory
        )
    
    def workflow_with_dilu_memory(self) -> List[str]:
        """
        Workflow using MemoryDILU (alternative memory strategy).
        
        DILU memory designed for task trajectory storage and retrieval.
        Alternative to MemoryGenerative - compare to see which works better.
        
        Returns:
            list: A ranked list of item IDs
        """
        huggingpt_planning = PlanningHUGGINGGPT(llm=self.llm)
        dilu_memory = MemoryDILU(llm=self.llm)
        return self._execute_generic_workflow(
            workflow_name="HuggingGPT + DILU Memory",
            planning_module=huggingpt_planning,
            reasoning_module=self.reasoning,
            memory_module=dilu_memory
        )
    
    def workflow_simple_efficient(self) -> List[str]:
        """
        Minimal workflow using only ReasoningIO (basic reasoning).
        
        Absolute simplest approach - no planning, no memory, just basic reasoning.
        Good for baseline comparison and speed testing.
        Fastest and cheapest option.
        
        Returns:
            list: A ranked list of item IDs
        """
        io_reasoning = ReasoningIO(
            profile_type_prompt='You are an intelligent recommendation system.',
            memory=None,
            llm=self.llm
        )
        return self._execute_generic_workflow(
            workflow_name="Simple Efficient (IO only)",
            planning_module=None,
            reasoning_module=io_reasoning,
            memory_module=None
        )
    
    def workflow_tot_with_memory(self) -> List[str]:
        """
        Advanced workflow combining Tree of Thoughts with Trajectory Planning memory.
        
        Most sophisticated combination:
        - ReasoningTOT: Generates 3 paths + 5 votes (8 API calls)
        - MemoryTP: Trajectory-based planning from past experiences
        
        ⚠️ WARNING: VERY EXPENSIVE (8+ API calls per task)
        Only use for research/benchmarking or critical high-value tasks.
        
        Returns:
            list: A ranked list of item IDs
        """
        tp_memory = MemoryTP(llm=self.llm)
        tot_reasoning = ReasoningTOT(
            profile_type_prompt='You are an intelligent recommendation system.',
            memory=tp_memory,
            llm=self.llm
        )
        return self._execute_generic_workflow(
            workflow_name="TOT + TP Memory (VERY EXPENSIVE)",
            planning_module=None,
            reasoning_module=tot_reasoning,
            memory_module=tp_memory
        )
    
    def workflow_deps_self_refine(self) -> List[str]:
        """
        Workflow combining Multi-Hop decomposition with iterative refinement.
        
        - PlanningDEPS: Multi-hop task decomposition
        - ReasoningSelfRefine: Initial output + reflection + refinement
        - MemoryGenerative: Importance-scored memory retrieval
        
        Good for complex tasks benefiting from both decomposition and refinement.
        Expensive but should produce high-quality results.
        
        Returns:
            list: A ranked list of item IDs
        """
        deps_planning = PlanningDEPS(llm=self.llm)
        generative_memory = MemoryGenerative(llm=self.llm)
        self_refine_reasoning = ReasoningSelfRefine(
            profile_type_prompt='You are an intelligent recommendation system.',
            memory=generative_memory,
            llm=self.llm
        )
        return self._execute_generic_workflow(
            workflow_name="DEPS + Self-Refine + Memory",
            planning_module=deps_planning,
            reasoning_module=self_refine_reasoning,
            memory_module=generative_memory
        )


if __name__ == "__main__":
    """
    Example usage of the Enhanced Recommendation Agent
    """
    # Configuration
    task_set = "yelp"  # Options: "amazon", "goodreads", "yelp"
    data_dir = "./data_processed"  # Update with your processed data directory
    
    # Initialize Simulator
    logging.info("Initializing simulator...")
    simulator = Simulator(
        data_dir=data_dir,
        device="auto",  # or "gpu" if you have GPU support
        cache=True  # Use cache for memory efficiency
    )
    
    # Load tasks and ground truth
    simulator.set_task_and_groundtruth(
        task_dir=f"./track2/{task_set}/tasks",
        groundtruth_dir=f"./track2/{task_set}/groundtruth"
    )
    
    # Set the enhanced agent
    simulator.set_agent(EnhancedRecommendationAgent)
    
    # Set LLM client - replace with your API key
    # You can use InfinigenceLLM or implement your own LLM client
    simulator.set_llm(InfinigenceLLM(api_key="your_api_key_here"))
    
    # Run simulation
    # Start with a small number to test, then run all tasks
    logging.info("Starting recommendation simulation...")
    outputs = simulator.run_simulation(
        number_of_tasks=10,  # Set to None to run all tasks
        enable_threading=True,
        max_workers=5  # Adjust based on your API rate limits
    )
    
    # Evaluate the agent
    logging.info("Evaluating results...")
    evaluation_results = simulator.evaluate()
    
    # Save results
    output_file = f'./evaluation_results_enhanced_track2_{task_set}.json'
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    logging.info(f"Evaluation complete! Results saved to {output_file}")
    logging.info(f"Results: {evaluation_results}")
    
    # Get detailed evaluation history if needed
    # evaluation_history = simulator.get_evaluation_history()

