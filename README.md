# 🕵️ MythBuster AI

MythBuster AI is an intelligent assistant that investigates myths, rumors, and common beliefs to determine whether they are **BUSTED**, **PLAUSIBLE**, or **CONFIRMED** using vector memory and real-time web search.

## 🚀 Features

- 🔍 **Semantic Search** using FAISS and HuggingFace embeddings (`all-MiniLM-L6-v2`)
- 🌐 **Real-time Web Search** with DuckDuckGo for recent or unknown claims
- 🧠 **Vector Store Memory** to retain and reuse learned evidence
- 🤖 **LLM Verdict Generation** via Groq API using LLaMA 3
- 🎨 **Optional Funny Image Generation** using Hugging Face Inference API (black-forest-labs/FLUX.1-dev)
- 🧰 **Gradio Interface** for an interactive, user-friendly chatbot
- 📜 **Logging** of claims, verdicts, and behavior for transparency

## 🛠️ Tech Stack

- **Python**
- **Gradio** (UI)
- **LangChain** (tool integration, document handling)
- **FAISS** (vector search)
- **DuckDuckGoSearchResults** (web fallback)
- **HuggingFace Embeddings**
- **Groq API** (LLaMA 3 model)
- **Hugging Face Inference API** (optional funny image generation)
- **Dotenv**(for secure API key management)

## 📦 Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/prshanthreddy/mythbuster.git
   cd mythbuster
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**

   Create a `.env` file in the root directory with the following content:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   HF_API_TOKEN=your_huggingface_api_token_here
   ```

5. **Run the App**

   ```bash
   python app.py
   ```

6. **Access the Interface**

   Once launched, the app will open in your default browser or provide a public link if `share=True`.

## ✨ Example Myths to Try

- “Drinking cold water causes a sore throat”
- “Humans only use 10% of their brain”
- “Goldfish have a 3-second memory”
- “Lightning never strikes the same place twice”

## 📁 Project Structure

```
.
├── assistant.log          # Logging output
├── app.py                 # Main application code
├── faiss_index/           # Stored vector memory
├── .env                   # Environment variable file (not committed)
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## 🧠 How It Works

1. **Input Claim** → via Gradio chatbot
2. **Search Vector Store** → looks for semantically similar prior evidence
3. **If Memory Vague or Missing** → uses DuckDuckGo tool to fetch relevant web results
4. **Prompt LLM** → send claim + context to Groq LLaMA 3 API
5. **Verdict Returned** → BUSTED / PLAUSIBLE / CONFIRMED with reasoning
6. **Optional Funny Image Generation** → if enabled via checkbox, uses Groq to generate a humorous image prompt, then calls Hugging Face’s `black-forest-labs/FLUX.1-dev` model to generate a matching image
7. **Log + Store Evidence** → saves new info into FAISS vector DB

## 🔐 Security

Make sure not to share your `.env` file or any API keys. Keep your `GROQ_API_KEY` and `HF_API_TOKEN` secure and never commit them to version control.
