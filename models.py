"""Data models for MythBuster AI."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from datetime import datetime

class Verdict(Enum):
    """Possible myth verdicts."""
    BUSTED = "BUSTED"
    PLAUSIBLE = "PLAUSIBLE" 
    CONFIRMED = "CONFIRMED"
    ERROR = "ERROR"

@dataclass
class MythResult:
    """Result of myth analysis."""
    claim: str
    verdict: Verdict
    reasoning: str
    source: str  # "memory" or "web"
    confidence: Optional[float] = None
    sources: Optional[List[str]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def format_response(self) -> str:
        """Format the result for display."""
        source_emoji = "🧠" if self.source == "memory" else "🌐"
        verdict_emoji = {
            Verdict.BUSTED: "❌",
            Verdict.PLAUSIBLE: "❓", 
            Verdict.CONFIRMED: "✅",
            Verdict.ERROR: "⚠️"
        }.get(self.verdict, "❓")
        
        header = f"{source_emoji} [{self.source.title()} Verdict] {verdict_emoji}\n\n"
        return f"{header}{self.reasoning}"

@dataclass 
class SearchResult:
    """Web search result."""
    title: str
    content: str
    url: Optional[str] = None
    
@dataclass
class ImageGenerationRequest:
    """Request for image generation."""
    myth: str
    prompt: Optional[str] = None
    enabled: bool = True