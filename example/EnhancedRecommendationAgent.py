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
    PlanningVoyager, PlanningIO, PlanningOPENAGI, PlanningHUGGINGGPT, PlanningTD
)
from websocietysimulator.agent.modules.reasoning_modules import (
    ReasoningCOT, ReasoningStepBack, ReasoningSelfRefine, ReasoningCOTSC, ReasoningTOT
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
        try:
            logging.info("Using Voyager Planning approach")
            
            # Initialize Voyager planning
            voyager_planning = PlanningVoyager(llm=self.llm)
            
            # Create a task description for the planner
            plan_task = f"Create recommendations for user {self.task['user_id']} from {len(self.task['candidate_list'])} items"
            
            # Generate plan using Voyager
            plan = voyager_planning(
                task_type='Recommendation',
                task_description=plan_task,
                feedback='',
                few_shot='sub-task 1: {"description": "Get user preferences", "reasoning instruction": "Analyze user history"}'
            )
            
            logging.info(f"Generated Voyager plan with {len(plan)} subtasks")
            
            # Execute the plan and gather information
            user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
            user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            candidate_items = []
            for item_id in self.task['candidate_list']:
                try:
                    item = self.interaction_tool.get_item(item_id=item_id)
                    if item:
                        candidate_items.append(self._filter_item_info(item))
                except Exception as e:
                    logging.warning(f"Error retrieving item {item_id}: {e}")
                    candidate_items.append({'item_id': item_id})
            
            # Use reasoning to generate recommendations
            task_description = self._create_recommendation_prompt(
                user_info=user_info,
                user_reviews=user_reviews,
                candidate_items=candidate_items,
                user_preference_summary=self._analyze_user_preferences(user_reviews)
            )
            
            result = self.reasoning(task_description)
            ranked_list = self._parse_recommendation_result(result)
            validated_list = self._validate_recommendations(ranked_list, self.task['candidate_list'])
            
            logging.info(f"Voyager workflow generated {len(validated_list)} recommendations")
            return validated_list
            
        except Exception as e:
            logging.error(f"Error in Voyager workflow: {e}", exc_info=True)
            return self.task['candidate_list']
    
    def workflow_with_self_refine(self) -> List[str]:
        """
        Alternative workflow using ReasoningSelfRefine module.
        
        Self-refine generates an initial recommendation, then reflects and refines it.
        Good for improving recommendation quality through iteration.
        
        Returns:
            list: A ranked list of item IDs
        """
        try:
            logging.info("Using Self-Refine Reasoning approach")
            
            # Initialize self-refine reasoning
            self_refine_reasoning = ReasoningSelfRefine(
                profile_type_prompt='You are an intelligent recommendation system.',
                memory=None,
                llm=self.llm
            )
            
            # Gather information (simplified for efficiency)
            user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
            user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            candidate_items = []
            for item_id in self.task['candidate_list']:
                try:
                    item = self.interaction_tool.get_item(item_id=item_id)
                    if item:
                        candidate_items.append(self._filter_item_info(item))
                except Exception as e:
                    candidate_items.append({'item_id': item_id})
            
            # Create task description
            task_description = self._create_recommendation_prompt(
                user_info=user_info,
                user_reviews=user_reviews,
                candidate_items=candidate_items,
                user_preference_summary=self._analyze_user_preferences(user_reviews)
            )
            
            # Use self-refine reasoning (generates and refines)
            result = self_refine_reasoning(task_description)
            
            ranked_list = self._parse_recommendation_result(result)
            validated_list = self._validate_recommendations(ranked_list, self.task['candidate_list'])
            
            logging.info(f"Self-Refine workflow generated {len(validated_list)} recommendations")
            return validated_list
            
        except Exception as e:
            logging.error(f"Error in Self-Refine workflow: {e}", exc_info=True)
            return self.task['candidate_list']
    
    def workflow_with_cot_sc(self) -> List[str]:
        """
        Alternative workflow using ReasoningCOTSC (Chain-of-Thought with Self-Consistency).
        
        Generates multiple reasoning paths and selects the most consistent answer.
        Good for improving reliability through consensus.
        Note: This uses more API calls (n=5) so it's more expensive.
        
        Returns:
            list: A ranked list of item IDs
        """
        try:
            logging.info("Using COT-SC (Self-Consistency) Reasoning approach")
            
            # Initialize COT-SC reasoning
            cot_sc_reasoning = ReasoningCOTSC(
                profile_type_prompt='You are an intelligent recommendation system.',
                memory=None,
                llm=self.llm
            )
            
            # Gather information
            user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
            user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            candidate_items = []
            for item_id in self.task['candidate_list']:
                try:
                    item = self.interaction_tool.get_item(item_id=item_id)
                    if item:
                        candidate_items.append(self._filter_item_info(item))
                except Exception as e:
                    candidate_items.append({'item_id': item_id})
            
            # Create task description
            task_description = self._create_recommendation_prompt(
                user_info=user_info,
                user_reviews=user_reviews,
                candidate_items=candidate_items,
                user_preference_summary=self._analyze_user_preferences(user_reviews)
            )
            
            # Use COT-SC reasoning (generates 5 answers and picks most common)
            result = cot_sc_reasoning(task_description)
            
            ranked_list = self._parse_recommendation_result(result)
            validated_list = self._validate_recommendations(ranked_list, self.task['candidate_list'])
            
            logging.info(f"COT-SC workflow generated {len(validated_list)} recommendations")
            return validated_list
            
        except Exception as e:
            logging.error(f"Error in COT-SC workflow: {e}", exc_info=True)
            return self.task['candidate_list']
    
    def workflow_with_voyager_memory(self) -> List[str]:
        """
        Alternative workflow using MemoryVoyager module.
        
        MemoryVoyager summarizes trajectories before storing them, providing
        concise memory retrieval. Good for learning from past recommendations.
        
        Returns:
            list: A ranked list of item IDs
        """
        try:
            logging.info("Using Voyager Memory approach")
            
            # Initialize Voyager memory
            voyager_memory = MemoryVoyager(llm=self.llm)
            
            # Gather user reviews and store in memory
            user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            if user_reviews:
                for review in user_reviews[:20]:
                    if 'text' in review and review['text']:
                        review_summary = f"Stars: {review.get('stars', 'N/A')}, Text: {review['text'][:200]}"
                        voyager_memory(f"review: {review_summary}")
            
            # Retrieve relevant context
            relevant_context = ""
            if user_reviews and len(user_reviews) > 0:
                sample_review = user_reviews[0].get('text', '')
                if sample_review:
                    relevant_context = voyager_memory(sample_review[:200])
            
            # Gather other information
            user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
            
            candidate_items = []
            for item_id in self.task['candidate_list']:
                try:
                    item = self.interaction_tool.get_item(item_id=item_id)
                    if item:
                        candidate_items.append(self._filter_item_info(item))
                except Exception as e:
                    candidate_items.append({'item_id': item_id})
            
            # Create enhanced task description with memory context
            user_preference_summary = self._analyze_user_preferences(user_reviews)
            if relevant_context:
                user_preference_summary += f"\n\nRelevant Memory Context:\n{relevant_context}"
            
            task_description = self._create_recommendation_prompt(
                user_info=user_info,
                user_reviews=user_reviews,
                candidate_items=candidate_items,
                user_preference_summary=user_preference_summary
            )
            
            result = self.reasoning(task_description)
            ranked_list = self._parse_recommendation_result(result)
            validated_list = self._validate_recommendations(ranked_list, self.task['candidate_list'])
            
            logging.info(f"Voyager Memory workflow generated {len(validated_list)} recommendations")
            return validated_list
            
        except Exception as e:
            logging.error(f"Error in Voyager Memory workflow: {e}", exc_info=True)
            return self.task['candidate_list']
    
    def workflow_with_openagi_planning(self) -> List[str]:
        """
        Alternative workflow using PlanningOPENAGI module.
        
        OpenAGI planning creates concise todo lists with minimal, relevant tasks.
        Good for efficient, streamlined recommendation generation.
        
        Returns:
            list: A ranked list of item IDs
        """
        try:
            logging.info("Using OpenAGI Planning approach")
            
            # Initialize OpenAGI planning
            openagi_planning = PlanningOPENAGI(llm=self.llm)
            
            # Create plan with OpenAGI
            plan_task = f"Generate personalized recommendations for user {self.task['user_id']}"
            plan = openagi_planning(
                task_type='Recommendation',
                task_description=plan_task,
                feedback='',
                few_shot='sub-task 1: {"description": "Analyze user", "reasoning instruction": "Get preferences"}'
            )
            
            logging.info(f"Generated OpenAGI plan with {len(plan)} subtasks")
            
            # Execute minimal information gathering
            user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
            user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            candidate_items = []
            for item_id in self.task['candidate_list']:
                try:
                    item = self.interaction_tool.get_item(item_id=item_id)
                    if item:
                        candidate_items.append(self._filter_item_info(item))
                except Exception as e:
                    candidate_items.append({'item_id': item_id})
            
            # Generate recommendations
            task_description = self._create_recommendation_prompt(
                user_info=user_info,
                user_reviews=user_reviews,
                candidate_items=candidate_items,
                user_preference_summary=self._analyze_user_preferences(user_reviews)
            )
            
            result = self.reasoning(task_description)
            ranked_list = self._parse_recommendation_result(result)
            validated_list = self._validate_recommendations(ranked_list, self.task['candidate_list'])
            
            logging.info(f"OpenAGI workflow generated {len(validated_list)} recommendations")
            return validated_list
            
        except Exception as e:
            logging.error(f"Error in OpenAGI workflow: {e}", exc_info=True)
            return self.task['candidate_list']
    
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
        try:
            logging.info("Using Hybrid Advanced approach (HuggingGPT + TP Memory + COT)")
            
            # 1. Initialize all modules
            huggingpt_planning = PlanningHUGGINGGPT(llm=self.llm)
            tp_memory = MemoryTP(llm=self.llm)
            cot_reasoning = ReasoningCOT(
                profile_type_prompt='You are an intelligent recommendation system.',
                memory=tp_memory,
                llm=self.llm
            )
            
            # 2. Create dependency-aware plan
            plan_task = f"Recommend items for user {self.task['user_id']} considering dependencies between data gathering and analysis"
            plan = huggingpt_planning(
                task_type='Recommendation',
                task_description=plan_task,
                feedback='',
                few_shot='sub-task 1: {"description": "Get user data", "reasoning instruction": "Must complete before matching"}'
            )
            
            logging.info(f"Generated HuggingGPT plan with {len(plan)} subtasks")
            
            # 3. Gather information and store in trajectory memory
            user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
            user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            # Store review patterns in trajectory memory
            if user_reviews:
                for review in user_reviews[:15]:
                    if 'text' in review and review['text']:
                        review_trajectory = f"User rated {review.get('stars', 'N/A')} stars: {review['text'][:150]}"
                        tp_memory(f"review: {review_trajectory}")
            
            # 4. Retrieve trajectory-based plans from memory
            memory_context = ""
            if user_reviews and len(user_reviews) > 0:
                sample_review = user_reviews[0].get('text', '')
                if sample_review:
                    memory_context = tp_memory(sample_review[:200])
            
            # 5. Gather candidate items
            candidate_items = []
            for item_id in self.task['candidate_list']:
                try:
                    item = self.interaction_tool.get_item(item_id=item_id)
                    if item:
                        candidate_items.append(self._filter_item_info(item))
                except Exception as e:
                    candidate_items.append({'item_id': item_id})
            
            # 6. Create comprehensive task description with memory insights
            user_preference_summary = self._analyze_user_preferences(user_reviews)
            if memory_context:
                user_preference_summary += f"\n\nTrajectory-Based Insights:\n{memory_context}"
            
            task_description = self._create_recommendation_prompt(
                user_info=user_info,
                user_reviews=user_reviews,
                candidate_items=candidate_items,
                user_preference_summary=user_preference_summary
            )
            
            # 7. Use COT reasoning for step-by-step recommendation
            result = cot_reasoning(task_description)
            
            # 8. Parse and validate
            ranked_list = self._parse_recommendation_result(result)
            validated_list = self._validate_recommendations(ranked_list, self.task['candidate_list'])
            
            logging.info(f"Hybrid Advanced workflow generated {len(validated_list)} recommendations")
            return validated_list
            
        except Exception as e:
            logging.error(f"Error in Hybrid Advanced workflow: {e}", exc_info=True)
            return self.task['candidate_list']


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

