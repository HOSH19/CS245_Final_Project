"""
Enhanced Simulation Agent using Planning, Reasoning, and Memory Modules

This agent demonstrates proper usage of the modular architecture available in the
websocietysimulator framework for Track 1 (User Behavior Simulation).
"""

import json
import logging
from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningIO, PlanningVoyager
from websocietysimulator.agent.modules.reasoning_modules import ReasoningCOT, ReasoningSelfRefine
from websocietysimulator.agent.modules.memory_modules import MemoryDILU, MemoryGenerative

logging.basicConfig(level=logging.INFO)


class EnhancedPlanning(PlanningIO):
    """
    Custom planning module that creates a structured plan for user behavior simulation.
    Inherits from PlanningIO for basic planning capabilities.
    """
    
    def __init__(self, llm):
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """
        Create a planning prompt specifically for user behavior simulation.
        
        The plan will include steps to:
        1. Gather user profile information
        2. Gather item/business information
        3. Retrieve relevant reviews from other users
        4. Retrieve user's own review history
        5. Synthesize all information to generate review
        """
        if feedback == '':
            prompt = '''You are a planning agent for a user behavior simulation task. 
Your goal is to create a systematic plan to gather all necessary information before simulating a user's review.

Here are example plans:
sub-task 1: {"description": "Retrieve the target user's profile and historical behavior", "reasoning instruction": "Understand the user's preferences, writing style, and rating patterns"}
sub-task 2: {"description": "Retrieve the target item/business details", "reasoning instruction": "Understand what the business offers and its characteristics"}
sub-task 3: {"description": "Retrieve reviews from other users about this item", "reasoning instruction": "Understand what others think and identify key aspects to comment on"}
sub-task 4: {"description": "Retrieve the user's own review history", "reasoning instruction": "Identify the user's personal writing style and rating patterns"}
sub-task 5: {"description": "Synthesize all information to generate an authentic review", "reasoning instruction": "Generate a review that matches the user's style while being relevant to the business"}

Now create a plan for this task:
Task Type: {task_type}
Task Description: {task_description}

Output your plan as a series of sub-tasks in the same format as the example above.
'''
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = f'''You are a planning agent for a user behavior simulation task.
Based on the following feedback, adjust your planning strategy:

Feedback: {feedback}

Task Type: {task_type}
Task Description: {task_description}

Create an improved plan that addresses the feedback.
'''
        return prompt


class EnhancedReasoning(ReasoningCOT):
    """
    Custom reasoning module that uses Chain-of-Thought reasoning
    to generate realistic user reviews.
    """
    
    def __init__(self, profile_type_prompt, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
    
    def __call__(self, task_description: str, feedback: str = ''):
        """
        Generate reasoning with detailed step-by-step analysis.
        """
        prompt = '''You are simulating a real user on a review platform. 
Think step-by-step to generate an authentic review.

Step 1: Analyze the user's profile and historical patterns
Step 2: Consider the business characteristics
Step 3: Reflect on what other reviewers have said
Step 4: Match your response to the user's writing style
Step 5: Generate appropriate rating and review text

Task Description:
{task_description}

Let's think through this step by step, then provide your final output in this format:
stars: [rating as float: 1.0, 2.0, 3.0, 4.0, or 5.0]
review: [your review text, 2-4 sentences]
'''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.7,  # Slightly higher temperature for more natural reviews
            max_tokens=1500
        )
        
        return reasoning_result


class EnhancedSimulationAgent(SimulationAgent):
    """
    Enhanced Simulation Agent that demonstrates proper usage of
    Planning, Reasoning, and Memory modules.
    
    This agent:
    1. Uses a planning module to create a structured information gathering plan
    2. Uses a reasoning module with chain-of-thought for review generation
    3. Uses memory to store and retrieve relevant reviews for better context
    """
    
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        
        # Initialize planning module
        self.planning = EnhancedPlanning(llm=self.llm)
        
        # Initialize reasoning module with COT
        self.reasoning = EnhancedReasoning(
            profile_type_prompt='You are a realistic user on a review platform.',
            llm=self.llm
        )
        
        # Initialize memory module for storing and retrieving relevant reviews
        self.memory = MemoryDILU(llm=self.llm)
        
        logging.info("EnhancedSimulationAgent initialized with Planning, Reasoning, and Memory modules")
    
    def workflow(self):
        """
        Main workflow for simulating user behavior.
        
        Returns:
            dict: Contains 'stars' (float) and 'review' (str)
        """
        try:
            # Step 1: Create a plan for information gathering
            logging.info(f"Creating plan for user {self.task['user_id']} and item {self.task['item_id']}")
            
            # For efficiency, we can use a predefined plan or generate one dynamically
            # Here we use a simple predefined plan to save API calls
            plan = [
                {
                    'description': 'Retrieve the target user profile and behavior',
                    'reasoning instruction': 'Understand user preferences and patterns'
                },
                {
                    'description': 'Retrieve the target business information',
                    'reasoning instruction': 'Understand business characteristics'
                },
                {
                    'description': 'Retrieve relevant reviews from other users',
                    'reasoning instruction': 'Learn what others think about this business'
                },
                {
                    'description': 'Retrieve user own review history',
                    'reasoning instruction': 'Match the user writing style'
                }
            ]
            
            # Step 2: Execute the plan - gather information
            user_info = None
            business_info = None
            other_reviews = []
            user_reviews = []
            
            for sub_task in plan:
                description = sub_task['description'].lower()
                
                if 'user profile' in description or 'user behavior' in description:
                    # Get user information
                    user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
                    logging.info(f"Retrieved user info for {self.task['user_id']}")
                
                elif 'business' in description or 'item' in description:
                    # Get business/item information
                    business_info = self.interaction_tool.get_item(item_id=self.task['item_id'])
                    logging.info(f"Retrieved business info for {self.task['item_id']}")
                
                elif 'other users' in description or 'relevant reviews' in description:
                    # Get reviews from other users about this business
                    reviews = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
                    if reviews:
                        # Store reviews in memory for similarity search
                        for review in reviews[:20]:  # Limit to top 20 reviews
                            if 'text' in review and review['text']:
                                self.memory(f"review: {review['text']}")
                                other_reviews.append(review)
                    logging.info(f"Retrieved {len(other_reviews)} reviews about the business")
                
                elif 'user.*review history' in description or 'own review' in description:
                    # Get user's own review history
                    user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
                    logging.info(f"Retrieved {len(user_reviews)} historical reviews from user")
            
            # Step 3: Use memory to find most relevant reviews
            relevant_context = ""
            if user_reviews and len(user_reviews) > 0:
                # Use the user's most recent review as a query to find similar content
                sample_review = user_reviews[0].get('text', '') if user_reviews[0].get('text') else ''
                if sample_review:
                    relevant_context = self.memory(sample_review[:200])  # Use first 200 chars as query
            
            # Step 4: Prepare context for reasoning
            # Summarize user info
            user_summary = self._summarize_user(user_info, user_reviews)
            
            # Summarize business info
            business_summary = self._summarize_business(business_info)
            
            # Summarize other reviews (get representative samples)
            other_reviews_summary = self._summarize_reviews(other_reviews, limit=5)
            
            # User's writing style from history
            user_style_summary = self._summarize_user_style(user_reviews, limit=3)
            
            # Step 5: Create task description for reasoning module
            task_description = f"""
You are simulating user behavior for a review platform.

USER PROFILE:
{user_summary}

BUSINESS/ITEM DETAILS:
{business_summary}

WHAT OTHERS ARE SAYING (sample reviews):
{other_reviews_summary}

USER'S WRITING STYLE (from history):
{user_style_summary}

RELEVANT CONTEXT FROM MEMORY:
{relevant_context if relevant_context else "No similar context found"}

YOUR TASK:
Based on all the information above, generate an authentic review that this user would write for this business.
Consider:
1. The user's rating patterns and preferences
2. The business's characteristics and what it offers
3. What other users have mentioned (positive/negative aspects)
4. The user's personal writing style and tone
5. Be authentic - users give 5 stars when delighted, 1-2 stars when disappointed

Output format:
stars: [1.0, 2.0, 3.0, 4.0, or 5.0]
review: [2-4 sentences matching the user's style]
"""
            
            # Step 6: Use reasoning module to generate the review
            logging.info("Generating review using reasoning module")
            result = self.reasoning(task_description)
            
            # Step 7: Parse the result
            stars, review_text = self._parse_result(result)
            
            # Ensure review length is within limits
            if len(review_text) > 512:
                review_text = review_text[:512]
            
            logging.info(f"Generated review with {stars} stars")
            
            return {
                "stars": stars,
                "review": review_text
            }
            
        except Exception as e:
            logging.error(f"Error in workflow: {e}", exc_info=True)
            # Return a reasonable default
            return {
                "stars": 3.0,
                "review": "Good experience overall."
            }
    
    def _summarize_user(self, user_info, user_reviews):
        """Summarize user information for the prompt."""
        if not user_info:
            return "User information not available."
        
        summary = f"User ID: {user_info.get('user_id', 'unknown')}\n"
        
        if 'name' in user_info:
            summary += f"Name: {user_info['name']}\n"
        
        if 'average_stars' in user_info or 'stars' in user_info:
            avg_stars = user_info.get('average_stars') or user_info.get('stars', 'N/A')
            summary += f"Average Rating Given: {avg_stars}\n"
        
        if 'review_count' in user_info:
            summary += f"Total Reviews Written: {user_info['review_count']}\n"
        
        if user_reviews:
            summary += f"Recent Review Count: {len(user_reviews)}"
        
        return summary
    
    def _summarize_business(self, business_info):
        """Summarize business/item information for the prompt."""
        if not business_info:
            return "Business information not available."
        
        summary = ""
        
        # Handle different platform formats
        if 'name' in business_info:
            summary += f"Name: {business_info['name']}\n"
        elif 'title' in business_info:
            summary += f"Title: {business_info['title']}\n"
        
        if 'stars' in business_info:
            summary += f"Average Rating: {business_info['stars']}\n"
        elif 'average_rating' in business_info:
            summary += f"Average Rating: {business_info['average_rating']}\n"
        
        if 'review_count' in business_info or 'rating_number' in business_info:
            count = business_info.get('review_count') or business_info.get('rating_number', 'N/A')
            summary += f"Number of Reviews: {count}\n"
        
        if 'categories' in business_info:
            summary += f"Categories: {business_info['categories']}\n"
        
        if 'attributes' in business_info and business_info['attributes']:
            summary += f"Attributes: {str(business_info['attributes'])[:200]}\n"
        
        if 'description' in business_info and business_info['description']:
            desc = business_info['description']
            summary += f"Description: {desc[:300]}...\n"
        
        return summary
    
    def _summarize_reviews(self, reviews, limit=5):
        """Get a sample of other users' reviews."""
        if not reviews:
            return "No reviews from other users available."
        
        summary = ""
        for i, review in enumerate(reviews[:limit]):
            if 'text' in review and review['text']:
                stars = review.get('stars', 'N/A')
                text = review['text'][:150]  # Limit length
                summary += f"{i+1}. ({stars} stars) {text}...\n"
        
        return summary
    
    def _summarize_user_style(self, user_reviews, limit=3):
        """Extract user's writing style from their review history."""
        if not user_reviews:
            return "No previous reviews from this user."
        
        summary = "User's previous reviews:\n"
        for i, review in enumerate(user_reviews[:limit]):
            if 'text' in review and review['text']:
                stars = review.get('stars', 'N/A')
                text = review['text'][:200]
                summary += f"{i+1}. ({stars} stars) {text}...\n"
        
        return summary
    
    def _parse_result(self, result):
        """Parse the LLM result to extract stars and review text."""
        try:
            # Look for lines containing 'stars:' and 'review:'
            lines = result.split('\n')
            
            stars = 3.0  # default
            review_text = "Good experience."  # default
            
            for line in lines:
                if 'stars:' in line.lower():
                    # Extract the number
                    parts = line.split(':')
                    if len(parts) > 1:
                        star_str = parts[1].strip()
                        # Remove any non-numeric characters except decimal point
                        star_str = ''.join(c for c in star_str if c.isdigit() or c == '.')
                        if star_str:
                            stars = float(star_str)
                            # Ensure it's a valid rating
                            stars = max(1.0, min(5.0, stars))
                            # Round to nearest valid value
                            stars = round(stars * 2) / 2  # Round to nearest 0.5
                            if stars not in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
                                stars = round(stars)  # Round to nearest integer
                
                elif 'review:' in line.lower():
                    # Extract the review text
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        review_text = parts[1].strip()
            
            # If review_text is still empty, try to extract any substantial text
            if review_text == "Good experience.":
                # Look for the longest sentence that's not a heading
                for line in lines:
                    if len(line) > 20 and ':' not in line[:20]:
                        review_text = line.strip()
                        break
            
            return stars, review_text
            
        except Exception as e:
            logging.error(f"Error parsing result: {e}")
            return 3.0, "Good experience overall."


if __name__ == "__main__":
    """
    Example usage of the Enhanced Simulation Agent
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
        task_dir=f"./track1/{task_set}/tasks",
        groundtruth_dir=f"./track1/{task_set}/groundtruth"
    )
    
    # Set the enhanced agent
    simulator.set_agent(EnhancedSimulationAgent)
    
    # Set LLM client - replace with your API key
    # You can use InfinigenceLLM or implement your own LLM client
    simulator.set_llm(InfinigenceLLM(api_key="your_api_key_here"))
    
    # Run simulation
    # Start with a small number to test, then run all tasks
    logging.info("Starting simulation...")
    outputs = simulator.run_simulation(
        number_of_tasks=10,  # Set to None to run all tasks
        enable_threading=True,
        max_workers=5  # Adjust based on your API rate limits
    )
    
    # Evaluate the agent
    logging.info("Evaluating results...")
    evaluation_results = simulator.evaluate()
    
    # Save results
    output_file = f'./evaluation_results_enhanced_track1_{task_set}.json'
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    logging.info(f"Evaluation complete! Results saved to {output_file}")
    logging.info(f"Results: {evaluation_results}")
    
    # Get detailed evaluation history if needed
    # evaluation_history = simulator.get_evaluation_history()

