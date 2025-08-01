"""Core services for MythBuster AI."""

import json
import requests
import asyncio
from pathlib import Path
from typing import Optional, List
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults

from config import config
from models import MythResult, Verdict, SearchResult, ImageGenerationRequest
from utils import is_vague_response, extract_verdict_from_response, sanitize_input, validate_claim
from logger import logger

class VectorStoreService:
    """Manages the vector store for knowledge persistence."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True
        )
        self._load_or_create_store()
    
    def _load_or_create_store(self):
        """Load existing vector store or create new one."""
        try:
            if config.vector_store_exists:
                logger.info("Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    config.vector_store_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            else:
                logger.info("Creating new vector store...")
                dummy_doc = Document(page_content="Initialization document")
                self.vector_store = FAISS.from_documents([dummy_doc], embedding=self.embeddings)
                self._reset_store()
        except Exception as e:
            logger.error(f"Error with vector store: {e}")
            # Fallback: create new store
            dummy_doc = Document(page_content="Initialization document")
            self.vector_store = FAISS.from_documents([dummy_doc], embedding=self.embeddings)
            self._reset_store()
    
    def _reset_store(self):
        """Reset vector store to empty state."""
        self.vector_store.index.reset()
        self.vector_store.docstore._dict.clear()
        self.vector_store.index_to_docstore_id.clear()
    
    def search_similar(self, query: str) -> List[Document]:
        """Search for similar documents in the vector store."""
        try:
            results = self.vector_store.similarity_search_with_score(
                query, k=config.max_search_results
            )
            # Filter by similarity threshold
            filtered = [doc for doc, score in results if score < config.similarity_threshold]
            return filtered
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def add_content(self, content: str) -> bool:
        """Add new content to the vector store."""
        try:
            # Check if content already exists
            existing = self.vector_store.similarity_search(content, k=3)
            if any(doc.page_content.strip() == content.strip() for doc in existing):
                logger.info("Content already exists in vector store")
                return False
            
            # Add new content
            new_doc = Document(page_content=content)
            chunks = self.splitter.split_documents([new_doc])
            self.vector_store.add_documents(chunks)
            self.save()
            logger.info("Added new content to vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding content to vector store: {e}")
            return False
    
    def save(self):
        """Save vector store to disk."""
        try:
            self.vector_store.save_local(config.vector_store_path)
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

class LLMService:
    """Handles interactions with the Groq LLM API."""
    
    def __init__(self):
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {config.groq_api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_myth(self, claim: str, context: str = None) -> MythResult:
        """Analyze a myth claim and return verdict."""
        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(claim, context)
            
            response = self._query_llm(system_prompt, user_prompt, config.llm_temperature)
            
            verdict = extract_verdict_from_response(response)
            source = "memory" if context else "web"
            
            return MythResult(
                claim=claim,
                verdict=verdict,
                reasoning=response,
                source=source
            )
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return MythResult(
                claim=claim,
                verdict=Verdict.ERROR,
                reasoning=f"Error analyzing claim: {str(e)}",
                source="error"
            )
    
    def generate_image_prompt(self, myth: str) -> str:
        """Generate a creative prompt for image generation."""
        try:
            system_prompt = (
                "You are a creative visual humorist. Given a myth or false belief, generate a funny, "
                "absurd, or satirical description of an image that visually illustrates the myth. "
                "Be specific, creative, and avoid using text in the image description. "
                "Keep it appropriate and humorous."
            )
            
            response = self._query_llm(system_prompt, myth, config.image_temperature)
            return response.strip()
        except Exception as e:
            logger.error(f"Image prompt generation error: {e}")
            return f"A humorous illustration of: {myth}"
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for myth analysis."""
        return (
            "You are MythBuster AI, an expert fact-checker and myth analyst. "
            "Your task is to analyze claims and determine their veracity. "
            "Always classify claims as one of: BUSTED (false), PLAUSIBLE (possible but uncertain), "
            "or CONFIRMED (true). Provide clear, factual reasoning for your verdict. "
            "Include sources when available. Be concise but thorough."
        )
    
    def _build_user_prompt(self, claim: str, context: str = None) -> str:
        """Build the user prompt with claim and context."""
        prompt = f'Claim: "{claim}"\n\n'
        
        if context:
            prompt += f"Relevant context and evidence:\n{context}\n\n"
        
        prompt += "Please analyze this claim and provide your verdict with reasoning."
        return prompt
    
    def _query_llm(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Query the LLM API."""
        payload = {
            "model": config.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 1000
        }
        
        response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
        response.raise_for_status()
        
        return response.json()['choices'][0]['message']['content']

class SearchService:
    """Handles web searches using DuckDuckGo."""
    
    def __init__(self):
        self.search_tool = DuckDuckGoSearchResults(max_results=5)
    
    @lru_cache(maxsize=100)
    def search(self, query: str) -> str:
        """Search the web for information about a query."""
        try:
            logger.info(f"Performing web search for: {query}")
            results = self.search_tool.run(query)
            return results
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return "Search unavailable due to rate limiting or connection issues."

class ImageService:
    """Handles image generation using Hugging Face API."""
    
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/" + config.image_model
        self.headers = {"Authorization": f"Bearer {config.hf_api_token}"}
    
    def generate_image(self, prompt: str, output_path: str = "funny_output.jpg") -> Optional[str]:
        """Generate an image from a text prompt."""
        if not config.has_image_generation:
            logger.warning("Image generation disabled: no HF token")
            return None
        
        try:
            logger.info(f"Generating image with prompt: {prompt[:100]}...")
            
            payload = {"inputs": prompt}
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json=payload, 
                timeout=60
            )
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Image saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return None

class MythBusterService:
    """Main service that orchestrates myth analysis."""
    
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.llm = LLMService()
        self.search = SearchService()
        self.image = ImageService()
    
    def analyze_claim(self, claim: str) -> MythResult:
        """Analyze a myth claim end-to-end."""
        # Sanitize and validate input
        claim = sanitize_input(claim)
        if not validate_claim(claim):
            return MythResult(
                claim=claim,
                verdict=Verdict.ERROR,
                reasoning="Invalid claim format. Please provide a clear, meaningful statement to fact-check.",
                source="validation"
            )
        
        logger.info(f"Analyzing claim: {claim}")
        
        # First, search vector store for existing knowledge
        relevant_docs = self.vector_store.search_similar(claim)
        
        if relevant_docs:
            # Use existing knowledge
            context = "\n\n".join(doc.page_content for doc in relevant_docs)
            result = self.llm.analyze_myth(claim, context)
            
            # If response is not vague, return it
            if not is_vague_response(result.reasoning):
                logger.info("Using knowledge from vector store")
                return result
        
        # Fallback to web search
        logger.info("Performing web search for additional information")
        search_results = self.search.search(claim)
        
        # Analyze with web search results
        result = self.llm.analyze_myth(claim, search_results)
        result.source = "web"
        
        # Store new knowledge in vector store
        if search_results and not is_vague_response(result.reasoning):
            self.vector_store.add_content(search_results)
        
        return result
    
    def generate_funny_image(self, request: ImageGenerationRequest) -> Optional[str]:
        """Generate a funny image for a myth."""
        if not request.enabled or not config.has_image_generation:
            return None
        
        try:
            # Generate creative prompt if not provided
            if not request.prompt:
                request.prompt = self.llm.generate_image_prompt(request.myth)
            
            # Generate image
            return self.image.generate_image(request.prompt)
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None