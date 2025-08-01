"""Utility functions for MythBuster AI."""

import re
import html
from typing import List
from models import Verdict

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent potential issues."""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags and decode HTML entities
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Limit length
    max_length = 500
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_vague_response(text: str) -> bool:
    """Check if a response is too vague to be useful."""
    if not text.strip():
        return True
        
    text_lower = text.lower()
    vague_phrases = [
        "i don't know", "not sure", "cannot answer", "no context", "not enough info",
        "uncertain", "please provide", "you haven't", "unknown", "not found",
        "unclear", "insufficient", "can't determine", "no information"
    ]
    
    return any(phrase in text_lower for phrase in vague_phrases)

def extract_verdict_from_response(response: str) -> Verdict:
    """Extract verdict from LLM response."""
    response_upper = response.upper()
    
    if "BUSTED" in response_upper:
        return Verdict.BUSTED
    elif "CONFIRMED" in response_upper:
        return Verdict.CONFIRMED
    elif "PLAUSIBLE" in response_upper:
        return Verdict.PLAUSIBLE
    else:
        return Verdict.PLAUSIBLE  # Default fallback

def format_search_results(results: str) -> List[str]:
    """Format search results for better readability."""
    if not results:
        return []
    
    # Split results into individual items
    items = results.split('\n')
    formatted = []
    
    for item in items:
        item = item.strip()
        if item and not item.startswith('['):
            formatted.append(item)
    
    return formatted[:5]  # Limit to top 5 results

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def validate_claim(claim: str) -> bool:
    """Validate that a claim is suitable for fact-checking."""
    if not claim or len(claim.strip()) < 5:
        return False
    
    # Check for obviously invalid claims
    invalid_patterns = [
        r'^[^a-zA-Z]*$',  # Only numbers/symbols
        r'^(.)\1{10,}',   # Repeated characters
    ]
    
    for pattern in invalid_patterns:
        if re.match(pattern, claim):
            return False
    
    return True