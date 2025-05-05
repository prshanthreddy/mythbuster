import os
import json
import requests
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing_extensions import List, TypedDict

import gradio as gr

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults

# ---------------- ENV & LOGGING ----------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["USER_AGENT"] = "my-custom-agent"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- EMBEDDINGS & VECTOR STORE ----------------

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

vector_store_path = "faiss_index"
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

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

# ---------------- GROQ LLM ----------------

def query_groq_llm(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",  # You can change to "mixtral-8x7b-32768" or others
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

# ---------------- DUCKDUCKGO TOOL ----------------

@tool
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo."""
    search = DuckDuckGoSearchResults()
    return search.run(query)

# ---------------- UTILS ----------------

def is_vague(text: str) -> bool:
    if not text.strip():
        return True
    vague_phrases = [
        "i don't know", "not sure", "cannot answer", "no context", "not enough info",
        "uncertain", "please provide", "you haven't", "unknown", "not found"
    ]
    return any(phrase in text.lower() for phrase in vague_phrases)

def is_realtime_query(text: str) -> bool:
    keywords = ["current", "today", "latest", "now", "who is", "trending", "new", "recent"]
    return any(k in text.lower() for k in keywords)

# ---------------- FALLBACK WEB SEARCH ----------------

def use_tool_only(question: str) -> str:
    logger.info(f"Real-time query detected: '{question}'")
    result = search_tool.invoke({"query": question})
    prompt = f"Here is information from the web:\n\n{result}\n\nAnswer this question: {question}"
    response = query_groq_llm(prompt)

    # Store new info if not already in memory
    new_doc = Document(page_content=result)
    chunks = splitter.split_documents([new_doc])
    existing = vector_store.similarity_search(result, k=5)
    already_exists = any(c.page_content.strip() == result.strip() for c in existing)

    if not already_exists:
        logger.info("Adding new content to vector store.")
        vector_store.add_documents(chunks)
        vector_store.save_local(vector_store_path)
    else:
        logger.info("Content already exists. Skipping add.")

    return f"[From Web Search]\n\n{response}"

# ---------------- ASK FUNCTION ----------------

def ask(question: str) -> str:
    logger.info(f"New Question: {question}")

    # Use FAISS with score
    retrieved_docs = vector_store.similarity_search_with_score(question, k=5)
    threshold = 0.25  # adjust as needed
    filtered_docs = [doc for doc, score in retrieved_docs if score < threshold]

    if not filtered_docs:
        logger.info("No relevant memory found. Using web search.")
        return use_tool_only(question)

    context = "\n\n".join(doc.page_content for doc in filtered_docs)
    prompt_text = f"""You are a helpful assistant.

Context:
{context}

Question: {question}"""

    response = query_groq_llm(prompt_text)

    if not is_vague(response):
        logger.info("Answered using memory.")
        return f"[From Memory]\n\n{response}"

    logger.info("Memory response vague. Using web search.")
    return use_tool_only(question)


# ---------------- GRADIO UI ----------------

with gr.Blocks(title="RAG + Groq LLM Assistant") as iface:
    gr.Markdown(
        """
        # 🤖 RAG + Real-Time AI Assistant (Groq + HuggingFace Embeddings)  
        Ask anything. The assistant uses memory or searches the web if needed.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="Assistant", height=400)
        with gr.Column(scale=1):
            msg = gr.Textbox(label="Your message", placeholder="Ask a question...", show_label=False)
            submit_btn = gr.Button("Submit")

    def user_message_handler(message, history):
        logger.info(f"User: {message}")
        response = ask(message)
        history.append((message, response))
        return "", history

    submit_btn.click(user_message_handler, [msg, chatbot], [msg, chatbot])
    msg.submit(user_message_handler, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    iface.launch(share=True)
