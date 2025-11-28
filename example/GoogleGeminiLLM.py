"""Google Gemini LLM client using GCP credits (no OpenAI needed)."""

from typing import Dict, List, Optional, Union
from websocietysimulator.llm import LLMBase
import google.generativeai as genai
import logging
import os
from dotenv import load_dotenv

# >>>>>>>>> NEW CODE START (1/2): Import time and random for retries >>>>>>>>>
import time
import random
# <<<<<<<<< NEW CODE END (1/2) <<<<<<<<<


try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    GOOGLE_EMBEDDINGS_AVAILABLE = False

load_dotenv()
logger = logging.getLogger("websocietysimulator")


class GoogleGeminiLLM(LLMBase):
    """Google Gemini LLM with embeddings - uses only Google APIs."""
    
    def __init__(
        self, 
        api_key: str = None,
        model: str = "gemini-2.0-flash",
        # >>>>>>>>> code modified by Emma (1/2) START : Use the newer embedding model >>>>>>>>>
        # Old: "models/embedding-001" (Prone to 429 errors)
        # New: "models/text-embedding-004" (Stable for paid accounts)
        embedding_model: str = "models/text-embedding-004"
        # <<<<<<<<< code modified by Emma END <<<<<<<<<
    ):
        """
        Initialize Google Gemini LLM.
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            model: "gemini-2.0-flash" (default), "gemini-1.5-flash", "gemini-1.5-pro"
            embedding_model: Google embedding model
        """
        super().__init__(model)
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google API key required. Get at: https://aistudio.google.com/app/apikey"
            )
        
        genai.configure(api_key=self.api_key)
        self.safety_settings = {
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }
        self.client = genai.GenerativeModel(model, safety_settings=self.safety_settings)
        
        # Initialize Google embeddings
        if GOOGLE_EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = GoogleGenerativeAIEmbeddings(
                    model=embedding_model, google_api_key=self.api_key
                )
                logger.info(f"✓ Google embeddings: {embedding_model}")
            except Exception as e:
                logger.error(f"Embedding init failed: {e}")
                self.embedding_model = None
        else:
            logger.error("Install: pip install langchain-google-genai")
            self.embedding_model = None
    
    # >>>>>>>>> code modified by Emma START (2/2): Replace the whole __call__ method >>>>>>>>>
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        
        """Call Gemini API with Retry Logic for 429 Errors."""
        
        # Retry parameters
        max_retries = 10
        base_delay = 2 
        
        for attempt in range(max_retries):
            try:
                # Original logic starts
                client = genai.GenerativeModel(model) if model and model != self.model else self.client
                
                gemini_messages = []
                system_instruction = None
                for msg in messages:
                    role, content = msg.get('role', 'user'), msg.get('content', '')
                    if role == 'system':
                        system_instruction = content
                    elif role == 'user':
                        gemini_messages.append({'role': 'user', 'parts': [content]})
                    elif role == 'assistant':
                        gemini_messages.append({'role': 'model', 'parts': [content]})
                
                config = genai.types.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
                
                if system_instruction:
                    temp_client = genai.GenerativeModel(
                        model or self.model, system_instruction=system_instruction,
                        safety_settings=self.safety_settings
                    )
                    response = temp_client.generate_content(
                        gemini_messages, generation_config=config, safety_settings=self.safety_settings
                    )
                elif len(gemini_messages) == 1 and gemini_messages[0]['role'] == 'user':
                    response = client.generate_content(
                        gemini_messages[0]['parts'][0], generation_config=config,
                        safety_settings=self.safety_settings
                    )
                else:
                    chat = client.start_chat(history=gemini_messages[:-1])
                    response = chat.send_message(
                        gemini_messages[-1]['parts'][0], generation_config=config,
                        safety_settings=self.safety_settings
                    )
                
                if response.candidates and response.candidates[0].content.parts:
                    return response.text if n == 1 else [response.text]
                else:
                    return "[Response blocked]" if n == 1 else ["[Response blocked]"]
                # Original logic ends

            except Exception as e:
                error_msg = str(e)
                # Handle 429 Rate Limit Errors with Exponential Backoff
                if "429" in error_msg or "Resource exhausted" in error_msg:
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                        print(f"⚠️  429 Limit Hit. Retrying in {sleep_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                        continue   # try again
                    else:
                        print(f"❌ Failed after {max_retries} retries.")
                        raise e 
                else:
                    # Other exceptions
                    print(f"Gemini API Error: {e}")
                    return ""
    # <<<<<<<<< code modified by Emma END <<<<<<<<<
    
    def get_embedding_model(self):
        """Get embedding model (GoogleGenerativeAIEmbeddings or None)."""
        if self.embedding_model is None:
            logger.warning("Install: pip install langchain-google-genai")
        return self.embedding_model


if __name__ == "__main__":
    # Quick test
    llm = GoogleGeminiLLM()
    print("✓ Gemini LLM initialized (gemini-2.0-flash)")
    response = llm(messages=[{"role": "user", "content": "Say hello!"}])
    print(f"Response: {response}")
    print(f"✓ Embeddings: {llm.get_embedding_model() is not None}")
