import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system."""

    # Model Configuration
    LLM_MODEL = "llama-3.1-8b-instant"  
    MODEL_PROVIDER = "groq"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default URLs (for document retrieval / RAG)
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment!")
        
        # Make sure key is in environment if needed
        os.environ["GROQ_API_KEY"] = api_key

        # Initialize Groq model with LangChain helper
        return init_chat_model(
            model=cls.LLM_MODEL,
            model_provider=cls.MODEL_PROVIDER,
            temperature=0.7
        )