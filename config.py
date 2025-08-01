"""Configuration management for MythBuster AI."""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Application configuration."""
    
    # API Keys
    groq_api_key: Optional[str] = None
    hf_api_token: Optional[str] = None
    
    # Model Configuration
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama3-8b-8192"
    image_model: str = "black-forest-labs/FLUX.1-dev"
    
    # Vector Store Configuration
    vector_store_path: str = "faiss_index"
    similarity_threshold: float = 0.5
    max_search_results: int = 5
    
    # Text Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # API Configuration
    llm_temperature: float = 0.7
    image_temperature: float = 1.0
    
    # UI Configuration
    share_gradio: bool = True
    server_port: int = 7860
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "assistant.log"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.hf_api_token = os.getenv("HF_API_TOKEN")
        
        # Set user agent for web searches
        os.environ["USER_AGENT"] = "MythBuster-AI/1.0"
        
        # Validate required API keys
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # HF token is optional for image generation
        if not self.hf_api_token:
            print("Warning: HF_API_TOKEN not provided. Image generation will be disabled.")
    
    @property
    def has_image_generation(self) -> bool:
        """Check if image generation is available."""
        return bool(self.hf_api_token)
    
    @property
    def vector_store_exists(self) -> bool:
        """Check if vector store exists."""
        return Path(self.vector_store_path).exists()

# Global configuration instance
config = Config()