"""
Custom LLM Client for Google Gemini API

This allows you to use your Google GCP credits with the AgentSociety Challenge framework.
"""

from typing import Dict, List, Optional, Union
from websocietysimulator.llm import LLMBase
from langchain_openai import OpenAIEmbeddings
import google.generativeai as genai
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("websocietysimulator")


class GoogleGeminiLLM(LLMBase):
    """
    LLM client for Google's Gemini API using your GCP credits.
    
    Setup Instructions:
    1. Go to https://aistudio.google.com/app/apikey
    2. Click "Create API Key" and select your GCP project
    3. Copy the API key
    4. Use it in your code or set as environment variable: GOOGLE_API_KEY
    """
    
    def __init__(
        self, 
        api_key: str = None,
        model: str = "gemini-1.5-flash",
        embedding_api_key: str = None
    ):
        """
        Initialize Google Gemini LLM
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY environment variable)
            model: Model name, options:
                - "gemini-1.5-flash" (recommended, faster and cheaper)
                - "gemini-1.5-pro" (more capable, more expensive)
                - "gemini-1.0-pro" (older, cheaper)
            embedding_api_key: Optional separate API key for embeddings (uses OpenAI by default)
        """
        super().__init__(model)
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google API key is required. Either pass api_key parameter or set GOOGLE_API_KEY environment variable.\n"
                "Get your key at: https://aistudio.google.com/app/apikey"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Configure safety settings to be less restrictive for simulation/research
        self.safety_settings = {
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }
        
        self.client = genai.GenerativeModel(
            model,
            safety_settings=self.safety_settings
        )
        
        # For embeddings, we'll use OpenAI embeddings (they're very good and cheap)
        # You can use OpenAI with a separate key, or Gemini embeddings
        self.embedding_api_key = embedding_api_key or os.getenv('OPENAI_API_KEY')
        if self.embedding_api_key:
            self.embedding_model = OpenAIEmbeddings(
                api_key=self.embedding_api_key,
                model="text-embedding-3-small"  # Very cheap, high quality
            )
        else:
            # Fallback: use a simple embedding (not recommended for production)
            logger.warning(
                "No embedding API key provided. Using basic embeddings. "
                "For better results, set OPENAI_API_KEY environment variable."
            )
            self.embedding_model = None
    
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        """
        Call Google Gemini API to get response
        
        Args:
            messages: List of messages with 'role' and 'content'
            model: Optional model override
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            stop_strs: Stop sequences (currently not supported by Gemini)
            n: Number of responses (if > 1, will call API n times)
            
        Returns:
            Union[str, List[str]]: Response text from LLM
        """
        try:
            # Use specified model or default
            if model and model != self.model:
                client = genai.GenerativeModel(model)
            else:
                client = self.client
            
            # Convert messages to Gemini format
            # Gemini uses a simpler format: just the content strings
            gemini_messages = []
            system_instruction = None
            
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    # Gemini handles system messages differently
                    system_instruction = content
                elif role == 'user':
                    gemini_messages.append({'role': 'user', 'parts': [content]})
                elif role == 'assistant':
                    gemini_messages.append({'role': 'model', 'parts': [content]})
            
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            # Handle multiple generations
            responses = []
            for _ in range(n):
                # If we have system instruction, create a new model with it
                if system_instruction:
                    temp_client = genai.GenerativeModel(
                        model or self.model,
                        system_instruction=system_instruction,
                        safety_settings=self.safety_settings
                    )
                    response = temp_client.generate_content(
                        gemini_messages,
                        generation_config=generation_config,
                        safety_settings=self.safety_settings
                    )
                else:
                    # For single user message (most common case)
                    if len(gemini_messages) == 1 and gemini_messages[0]['role'] == 'user':
                        response = client.generate_content(
                            gemini_messages[0]['parts'][0],
                            generation_config=generation_config,
                            safety_settings=self.safety_settings
                        )
                    else:
                        # For multi-turn conversations
                        chat = client.start_chat(history=gemini_messages[:-1])
                        response = chat.send_message(
                            gemini_messages[-1]['parts'][0],
                            generation_config=generation_config,
                            safety_settings=self.safety_settings
                        )
                
                # Handle blocked or empty responses
                if response.candidates and response.candidates[0].content.parts:
                    responses.append(response.text)
                else:
                    # Response was blocked or empty
                    finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                    safety_ratings = response.candidates[0].safety_ratings if response.candidates else []
                    
                    error_msg = f"Response blocked or empty. Finish reason: {finish_reason}"
                    if safety_ratings:
                        error_msg += f"\nSafety ratings: {[(r.category, r.probability) for r in safety_ratings]}"
                    
                    logger.warning(error_msg)
                    # Return a safe fallback response
                    responses.append("[Response blocked by safety filters. Please try rephrasing your request or adjusting safety settings.]")
            
            # Return single string or list based on n
            if n == 1:
                return responses[0]
            else:
                return responses
                
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            raise e
    
    def get_embedding_model(self):
        """
        Get the embedding model
        
        Returns:
            OpenAIEmbeddings or None
        """
        if self.embedding_model is None:
            logger.warning(
                "No embedding model available. Please provide OPENAI_API_KEY for embeddings."
            )
        return self.embedding_model


# Example usage
if __name__ == "__main__":
    """
    Example of how to use GoogleGeminiLLM with the enhanced agents
    """
    import os
    from websocietysimulator import Simulator
    from EnhancedSimulationAgent import EnhancedSimulationAgent
    
    # Set your Google API key
    # Get it from: https://aistudio.google.com/app/apikey
    google_api_key = os.getenv('GOOGLE_API_KEY', 'your-google-api-key-here')
    
    # Optional: Set OpenAI key for embeddings (cheap, high quality)
    # You can get OpenAI credits separately or use another embedding service
    openai_api_key = os.getenv('OPENAI_API_KEY', 'your-openai-key-for-embeddings')
    
    # Initialize Gemini LLM
    llm = GoogleGeminiLLM(
        api_key=google_api_key,
        model="gemini-1.5-flash",  # Fast and cheap, good for development
        embedding_api_key=openai_api_key
    )
    
    # Test the LLM
    print("Testing Gemini LLM...")
    test_response = llm(
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        temperature=0.7,
        max_tokens=50
    )
    print(f"Response: {test_response}")
    
    # Use with the simulator
    print("\nInitializing simulator with Gemini...")
    simulator = Simulator(data_dir="../data_processed", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir="./track1/yelp/tasks",
        groundtruth_dir="./track1/yelp/groundtruth"
    )
    
    simulator.set_agent(EnhancedSimulationAgent)
    simulator.set_llm(llm)
    
    # Run a small test
    print("Running simulation with 2 tasks...")
    outputs = simulator.run_simulation(number_of_tasks=2)
    results = simulator.evaluate()
    
    print(f"\nResults: {results}")

