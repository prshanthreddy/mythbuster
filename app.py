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
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults

# ---------------- ENV & LOGGING ----------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
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
    system_prompt = (
        "You are MythBuster AI. A user will state a myth or claim. "
        "Your task is to analyze the claim using the provided context or search result. "
        "Decide if the claim is BUSTED, PLAUSIBLE, or CONFIRMED. Justify your verdict briefly and factually.Provide Source if possible "
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
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def generate_funny_image_prompt(myth: str) -> str:
    system_prompt = (
        "You are a creative visual humorist. Given a myth or false belief, generate a funny or absurd description of an image "
        "that visually illustrates or mocks the myth. Be creative, specific, and avoid using text in the image."
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

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# ------------Hugging face model for image generation -------
def generate_image_from_prompt(prompt: str, api_token: str, output_path="funny_output.jpg") -> str:
    url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": prompt}

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path
# ---------------- DUCKDUCKGO TOOL ----------------

@tool
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        search = DuckDuckGoSearchResults()
        return search.run(query)
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return "No results due to error or rate limiting."


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
    logger.info(f"Real-time myth query detected: '{claim}'")
    result = search_tool.invoke({"query": claim})
    
    prompt = f"""
Claim: "{claim}"

Evidence from the Web:
{result}

Determine if the claim is BUSTED, PLAUSIBLE, or CONFIRMED. Explain briefly.
"""

    response = query_groq_llm(prompt)

    # Store result in memory
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
        return f"🧠 [Memory Verdict]\n\n{response} "

    return f"🌐 [Web Verdict]\n\n{response}"


# ---------------- ASK FUNCTION ----------------

def ask(claim: str) -> str:
    logger.info(f"New Claim: {claim}")
    retrieved_docs = vector_store.similarity_search_with_score(claim, k=5)
    threshold = 0.5
    filtered_docs = [doc for doc, score in retrieved_docs if score < threshold]

    if not filtered_docs:
        logger.info("No relevant memory. Using web search.")
        return use_tool_only(claim)

    context = "\n\n".join(doc.page_content for doc in filtered_docs)
    prompt = f"""
Claim: "{claim}"

Context from known sources:
{context}

Determine if the claim is BUSTED, PLAUSIBLE, or CONFIRMED. Explain briefly.
"""
    response = query_groq_llm(prompt)

    if not is_vague(response):
        logger.info("Myth verdict given from memory.")
        return f"🧠 [Memory Verdict]\n\n{response}"

    logger.info("Memory response vague. Falling back to web.")
    return use_tool_only(claim)


# ---------------- GRADIO UI ----------------

with gr.Blocks(title="MythBuster AI") as iface:
    gr.Markdown("""
    # 🕵️ MythBuster AI  
    **Ask me about any myth, rumor, or common belief — I'll investigate it and give you a verdict!**  
    💡 I classify myths as:
    - ✅ **CONFIRMED**
    - ❓ **PLAUSIBLE**
    - ❌ **BUSTED**
    """)
    gr.Markdown("## 🧠 Myth Verdicts")

    with gr.Row():
        chatbot = gr.Chatbot(label="🧠 Myth Verdicts", height=400, type="messages")
        funny_output = gr.Image(label="😂 Funny Image")

    with gr.Row():
        msg = gr.Textbox(
            label="Enter a myth or claim",
            placeholder="e.g., 'Drinking cold water causes a sore throat'",
            show_label=False
        )
        gen_image = gr.Checkbox(label="🎨 Generate Funny Image", value=True)
        submit_btn = gr.Button("🚀 Bust This Myth")

    def user_message_handler(message, history, generate_img):
        logger.info(f"User claim: {message}")
        if history is None:
            history = []

        try:
            response = ask(message)
        except Exception as e:
            response = f"❌ Error: {e}"
            logger.error(str(e))
            return "", history

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        image_path = None
        if generate_img:
            try:
                funny_prompt = generate_funny_image_prompt(message)
                image_path = generate_image_from_prompt(funny_prompt, HF_API_TOKEN)
            except Exception as e:
                logger.error(f"Image generation error: {e}")
                image_path = None

        return "", history, image_path

    submit_btn.click(user_message_handler, [msg, chatbot, gen_image], [msg, chatbot, funny_output])
    msg.submit(user_message_handler, [msg, chatbot, gen_image], [msg, chatbot, funny_output])

    gr.Examples(
        examples=[
            ["Drinking cold water causes a sore throat"],
            ["Humans only use 10% of their brain"],
            ["Goldfish have a 3-second memory"],
            ["You can see the Great Wall of China from space"],
            ["Eating carrots improves your eyesight"],
            ["Vaccines cause autism"],
            ["Bats are blind"],
            ["Lightning never strikes the same place twice"],
            ["Cracking your knuckles causes arthritis"],
            ["The Great Wall of China is visible from space"]
        ],
        inputs=msg,
        label="Examples"
    )

if __name__ == "__main__":
    iface.launch(share=True)
