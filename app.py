# mythbuster_streamlit.py

import os
import json
import requests
import logging
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults
from duckduckgo_search.exceptions import DuckDuckGoSearchException
import torch
import faiss

# ---------------- ENV & LOGGING ----------------

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # Avoid torch watcher crash
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
os.environ["USER_AGENT"] = "my-custom-agent"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- EMBEDDINGS & VECTOR STORE ----------------

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

vector_store_path = "faiss_index"
if Path(vector_store_path).exists():
    logger.info("\U0001F4C2 Loading existing vector store...")
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
else:
    logger.info("\U0001F4E6 Initializing new vector store...")
    dummy_doc = Document(page_content="Init doc")
    vector_store = FAISS.from_documents([dummy_doc], embedding=embeddings)
    vector_store.index.reset()
    vector_store.docstore._dict.clear()
    vector_store.index_to_docstore_id.clear()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

# ---------------- FUNCTION DEFINITIONS ----------------

def query_groq_llm(prompt: str) -> str:
    system_prompt = (
        "You are MythBuster AI. A user will state a myth or claim. "
        "Your task is to analyze the claim using the provided context or search result. "
        "Decide if the claim is BUSTED, PLAUSIBLE, or CONFIRMED. Justify your verdict briefly and factually."
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

def generate_image_from_prompt(prompt: str, api_token: str, output_path="funny_output.jpg", retries=3) -> str:
    url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": prompt}

    for attempt in range(retries):
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 504:
            logging.warning(f"Timeout. Retrying {attempt + 1}/{retries}...")
            continue
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path

    raise Exception("Image generation failed after retries.")

def use_tool_only(claim: str) -> tuple[str, str]:
    logger.info(f"Real-time myth query detected: '{claim}'")
    try:
        result = DuckDuckGoSearchResults().run(claim)
    except DuckDuckGoSearchException as e:
        return "DuckDuckGo search failed or timed out.", "web"

    prompt = f"""
Claim: "{claim}"

Evidence from the Web:
{result}

Determine if the claim is BUSTED, PLAUSIBLE, or CONFIRMED. Explain briefly.
"""
    response = query_groq_llm(prompt)
    new_doc = Document(page_content=result)
    chunks = splitter.split_documents([new_doc])
    existing = vector_store.similarity_search(result, k=5)
    already_exists = any(c.page_content.strip() == result.strip() for c in existing)

    if not already_exists:
        logger.info("Adding new content to vector store.")
        vector_store.add_documents(chunks)
        vector_store.save_local(vector_store_path)
    return response, "web"

def ask(claim: str) -> tuple[str, str]:
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
    return response, "memory"

# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="MythBuster AI", layout="centered")
st.title("\U0001F575 MythBuster AI")
st.markdown("Ask me about any myth, rumor, or belief. I’ll classify it as **BUSTED**, **PLAUSIBLE**, or **CONFIRMED**.")

st.markdown("### \U0001F4A1 Choose an example or enter your own claim")

example_claims = [
    "",
    "Drinking cold water causes a sore throat",
    "Humans only use 10% of their brain",
    "Goldfish have a 3-second memory",
    "You can see the Great Wall of China from space",
    "Eating carrots improves your eyesight",
    "Vaccines cause autism",
    "Bats are blind",
    "Lightning never strikes the same place twice",
    "Cracking your knuckles causes arthritis"
]

selected_example = st.selectbox("Examples", example_claims, index=0)
custom_input = st.text_input("\u270D\ufe0f Or type your own:", placeholder="e.g., 'Goldfish have a 3-second memory'")
generate_image = st.checkbox("\U0001F3A8 Generate Funny Image", value=True)

# Determine which input to use
final_claim = custom_input.strip() if custom_input.strip() else selected_example.strip()

if st.button("\U0001F680 Bust This Myth") and final_claim:
    with st.spinner("Analyzing myth..."):
        verdict, source = ask(final_claim)
        label = "\U0001F9E0 Memory Verdict" if source == "memory" else "\U0001F310 Web Verdict"
        st.markdown(f"### {label}\n{verdict}")

        if generate_image:
            with st.spinner("Generating image..."):
                prompt = generate_funny_image_prompt(final_claim)
                image_path = generate_image_from_prompt(prompt, HF_API_TOKEN)
                st.image(image_path, caption="Humorous Illustration")