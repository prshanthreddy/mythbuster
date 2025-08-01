import os
import json
import requests
import logging
import re
import html
from pathlib import Path
from dotenv import load_dotenv
from typing_extensions import List, TypedDict
from functools import lru_cache

import gradio as gr

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults

# ---------------- ENV & LOGGING ----------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
os.environ["USER_AGENT"] = "MythBuster-AI/1.0"

# Validate required API keys
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY environment variable is required. Please check your .env file.")

if not HF_API_TOKEN:
    print("⚠️  Warning: HF_API_TOKEN not provided. Image generation will be disabled.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mythbuster")

# ---------------- EMBEDDINGS & VECTOR STORE ----------------

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

try:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    logger.info("✅ Embedding model loaded successfully")
except Exception as e:
    logger.warning(f"⚠️  Warning: Could not load embedding model: {e}")
    logger.warning("Vector search functionality will be limited")
    embeddings = None

vector_store_path = "faiss_index"
vector_store = None

if embeddings:
    try:
        if Path(vector_store_path).exists():
            logger.info("📂 Loading existing vector store...")
            vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        else:
            logger.info("📦 Initializing new vector store...")
            dummy_doc = Document(page_content="Init doc")
            vector_store = FAISS.from_documents([dummy_doc], embedding=embeddings)
            vector_store.index.reset()
            vector_store.docstore._dict.clear()
            vector_store.index_to_docstore_id.clear()
        logger.info("✅ Vector store initialized")
    except Exception as e:
        logger.error(f"❌ Vector store initialization failed: {e}")
        vector_store = None
else:
    logger.warning("⚠️  Vector store disabled (no embeddings available)")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

# ---------------- UTILITY FUNCTIONS ----------------

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent potential issues."""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags and decode HTML entities
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Limit length to prevent abuse
    max_length = 500
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

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

# ---------------- GROQ LLM ----------------

def query_groq_llm(prompt: str) -> str:
    """Query Groq LLM with improved error handling."""
    system_prompt = (
        "You are MythBuster AI, an expert fact-checker and myth analyst. "
        "Your task is to analyze claims and determine their veracity. "
        "Always classify claims as one of: BUSTED (false), PLAUSIBLE (possible but uncertain), "
        "or CONFIRMED (true). Provide clear, factual reasoning for your verdict. "
        "Include sources when available. Be concise but thorough."
    )

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API error: {e}")
        return f"❌ API Error: Unable to process request. Please try again later."
    except Exception as e:
        logger.error(f"Unexpected error in LLM query: {e}")
        return f"❌ Unexpected error occurred. Please try again."


def generate_funny_image_prompt(myth: str) -> str:
    """Generate a creative prompt for image generation."""
    system_prompt = (
        "You are a creative visual humorist. Given a myth or false belief, generate a funny, "
        "absurd, or satirical description of an image that visually illustrates the myth. "
        "Be specific, creative, and avoid using text in the image description. "
        "Keep it appropriate and humorous."
    )

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": myth}
        ],
        "temperature": 1.0
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Image prompt generation error: {e}")
        return f"A humorous illustration of: {myth}"

# ---------------- IMAGE GENERATION ----------------

def generate_image_from_prompt(prompt: str, api_token: str, output_path="funny_output.jpg") -> str:
    """Generate image with improved error handling."""
    if not api_token:
        logger.warning("Image generation skipped: no HF token provided")
        return None
        
    url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": prompt}

    try:
        logger.info(f"Generating image with prompt: {prompt[:100]}...")
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Image saved to: {output_path}")
        return output_path
    except requests.exceptions.Timeout:
        logger.error("Image generation timed out")
        return None
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return None
# ---------------- DUCKDUCKGO TOOL WITH CACHING ----------------

@tool
@lru_cache(maxsize=100)
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo with caching."""
    try:
        logger.info(f"Performing web search for: {query}")
        search = DuckDuckGoSearchResults(max_results=5)
        return search.run(query)
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return "Search unavailable due to rate limiting or connection issues."


# ---------------- UTILS ----------------

def is_vague(text: str) -> bool:
    if not text.strip():
        return True
    vague_phrases = [
        "i don't know", "not sure", "cannot answer", "no context", "not enough info",
        "uncertain", "please provide", "you haven't", "unknown", "not found"
    ]
    return any(phrase in text.lower() for phrase in vague_phrases)


# ---------------- FALLBACK WEB SEARCH ----------------

def use_tool_only(claim: str) -> str:
    """Use web search when vector store is unavailable or insufficient."""
    logger.info(f"Using web search for: '{claim}'")
    result = search_tool.invoke({"query": claim})
    
    prompt = f"""
Claim: "{claim}"

Evidence from web search:
{result}

Analyze this claim and determine if it is BUSTED, PLAUSIBLE, or CONFIRMED. 
Provide clear reasoning based on the evidence.
"""

    response = query_groq_llm(prompt)

    # Try to store result in memory if vector store is available  
    if vector_store and result:
        try:
            new_doc = Document(page_content=result)
            chunks = splitter.split_documents([new_doc])
            existing = vector_store.similarity_search(result, k=5)
            already_exists = any(c.page_content.strip() == result.strip() for c in existing)

            if not already_exists:
                logger.info("Adding new content to vector store.")
                vector_store.add_documents(chunks)
                vector_store.save_local(vector_store_path)
            else:
                logger.info("Content already exists in vector store.")
        except Exception as e:
            logger.error(f"Failed to store search result: {e}")

    return f"🌐 **[Web Search Verdict]**\n\n{response}"


# ---------------- IMPROVED ASK FUNCTION ----------------

def ask(claim: str) -> str:
    """Enhanced myth analysis with better error handling."""
    # Sanitize and validate input
    original_claim = claim
    claim = sanitize_input(claim)
    
    if not validate_claim(claim):
        return "❌ **Invalid Input**: Please provide a clear, meaningful claim to fact-check (at least 5 characters)."
    
    logger.info(f"Analyzing claim: {claim}")
    
    try:
        # Search vector store for existing knowledge if available
        if vector_store:
            retrieved_docs = vector_store.similarity_search_with_score(claim, k=5)
            threshold = 0.5
            filtered_docs = [doc for doc, score in retrieved_docs if score < threshold]

            if filtered_docs:
                # Use existing knowledge
                context = "\n\n".join(doc.page_content for doc in filtered_docs)
                prompt = f"""
Claim: "{claim}"

Context from knowledge base:
{context}

Determine if the claim is BUSTED, PLAUSIBLE, or CONFIRMED. Explain briefly with evidence.
"""
                response = query_groq_llm(prompt)

                if not is_vague(response):
                    logger.info("Providing verdict from existing knowledge.")
                    return f"🧠 **[Memory Verdict]**\n\n{response}"

        # Fallback to web search
        logger.info("Using web search for analysis.")
        return use_tool_only(claim)
        
    except Exception as e:
        logger.error(f"Error in ask function: {e}")
        return f"❌ **Error**: Unable to analyze claim. Please try again. ({str(e)})"


# ---------------- ENHANCED GRADIO UI ----------------

# Custom CSS for better styling
custom_css = """
<style>
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.status-message {
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    text-align: center;
    font-weight: bold;
}
.loading { background-color: #e3f2fd; color: #1976d2; }
.success { background-color: #e8f5e8; color: #388e3c; }
.error { background-color: #ffebee; color: #d32f2f; }
</style>
"""

with gr.Blocks(
    title="🕵️ MythBuster AI - Enhanced", 
    theme=gr.themes.Soft(),
    css=custom_css
) as iface:
    
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>🕵️ MythBuster AI</h1>
            <p><strong>Intelligent Myth Detection & Fact-Checking Assistant</strong></p>
            <p>Ask me about any myth, rumor, or common belief — I'll investigate it using AI and real-time web search!</p>
        </div>
    """)
    
    # Status indicator
    status_display = gr.HTML(value="", visible=True)
    
    # Main content
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 🧠 Myth Analysis Results")
            chatbot = gr.Chatbot(
                label="Conversation History", 
                height=400, 
                type="messages",
                show_copy_button=True,
                container=True
            )
        
        with gr.Column(scale=1):
            gr.Markdown("## 🎨 Generated Image")
            funny_output = gr.Image(
                label="Humorous Visualization",
                height=300,
                show_label=True
            )
            
            # Quick info panel
            gr.HTML("""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center; margin-top: 1rem;">
                    <p><strong>🎯 Powered by AI</strong></p>
                    <p>LLaMA 3.1 + Vector Search</p>
                    <p>📚 Knowledge Base + 🌐 Web Search</p>
                </div>
            """)

    # Input section
    with gr.Row():
        with gr.Column(scale=4):
            msg = gr.Textbox(
                label="",
                placeholder="Enter a myth or claim to fact-check (e.g., 'Drinking cold water causes a sore throat')",
                show_label=False,
                lines=2,
                max_lines=3
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("🚀 Analyze Myth", variant="primary", size="lg")
    
    # Options
    with gr.Row():
        gen_image = gr.Checkbox(
            label="🎨 Generate funny image", 
            value=bool(HF_API_TOKEN),
            interactive=bool(HF_API_TOKEN)
        )
        clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
        
    if not HF_API_TOKEN:
        gr.HTML("<em>💡 Image generation disabled (HF_API_TOKEN not provided)</em>")

    def user_message_handler(message, history, generate_img):
        """Enhanced message handler with status updates."""
        if not message.strip():
            return "", history, None, ""

        logger.info(f"User claim: {message}")
        if history is None:
            history = []

        # Show loading status
        status_html = '<div class="status-message loading">🔍 Analyzing your claim... Please wait.</div>'
        
        try:
            # Process the claim
            response = ask(message)
            
            # Add to conversation history
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})

            # Generate image if requested
            image_path = None
            if generate_img and HF_API_TOKEN:
                try:
                    funny_prompt = generate_funny_image_prompt(message)  
                    image_path = generate_image_from_prompt(funny_prompt, HF_API_TOKEN)
                except Exception as e:
                    logger.error(f"Image generation error: {e}")

            # Success status
            status_html = '<div class="status-message success">✅ Analysis complete!</div>'
            
        except Exception as e:
            logger.error(f"Handler error: {e}")
            error_msg = f"❌ **Error**: {str(e)}\n\nPlease try again or rephrase your question."
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            status_html = '<div class="status-message error">❌ An error occurred. Please try again.</div>'
            image_path = None

        return "", history, image_path, status_html

    # Event handlers
    submit_btn.click(
        user_message_handler, 
        inputs=[msg, chatbot, gen_image], 
        outputs=[msg, chatbot, funny_output, status_display]
    )
    
    msg.submit(
        user_message_handler, 
        inputs=[msg, chatbot, gen_image], 
        outputs=[msg, chatbot, funny_output, status_display]
    )
    
    clear_btn.click(
        lambda: ([], None, ""),
        outputs=[chatbot, funny_output, status_display]
    )

    # Enhanced examples
    gr.Examples(
        examples=[
            ["Drinking cold water causes a sore throat"],
            ["Humans only use 10% of their brain"],
            ["Goldfish have a 3-second memory"],
            ["You can see the Great Wall of China from space"],
            ["Eating carrots improves your eyesight"],
            ["Lightning never strikes the same place twice"],
            ["Cracking your knuckles causes arthritis"],
            ["Bulls are enraged by the color red"],
            ["We only have five senses"],
            ["Hair and nails continue growing after death"]
        ],
        inputs=msg,
        label="💡 Example Myths to Try"
    )
    
    # Help section
    with gr.Accordion("ℹ️ How MythBuster AI Works", open=False):
        gr.Markdown("""
        ### 🔍 Analysis Process:
        1. **📝 Input Processing**: Your claim is sanitized and validated
        2. **🧠 Memory Search**: Check existing knowledge base for similar claims  
        3. **🌐 Web Research**: If needed, search current information online
        4. **🤖 AI Analysis**: Advanced language model analyzes evidence
        5. **📊 Verdict**: Classified as CONFIRMED ✅, PLAUSIBLE ❓, or BUSTED ❌
        6. **🎨 Visual Fun**: Optional humorous image generation
        
        ### 💡 Tips for Best Results:
        - Be specific and clear in your claims
        - Use complete sentences when possible  
        - Try rephrasing if you get an unclear answer
        - Check multiple similar claims for comparison
        
        ### 🔒 Privacy & Security:
        - Your conversations are processed securely
        - API keys are stored safely in environment variables
        - All interactions are logged for quality improvement only
        """)

if __name__ == "__main__":
    logger.info("🚀 Starting MythBuster AI...")
    logger.info(f"Image generation available: {bool(HF_API_TOKEN)}")
    iface.launch(share=True)
