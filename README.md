# 🕵️ MythBuster AI - Enhanced

MythBuster AI is an intelligent assistant that investigates myths, rumors, and common beliefs to determine whether they are **BUSTED**, **PLAUSIBLE**, or **CONFIRMED** using vector memory and real-time web search.

## ✨ What's New in v2.0

### 🚀 Enhanced Features
- **🛡️ Robust Error Handling**: Graceful fallbacks when APIs are unavailable
- **🎨 Improved UI**: Better styling, loading indicators, and user feedback  
- **🔒 Input Validation**: Sanitization and validation to prevent issues
- **📱 Offline Resilience**: Works even without internet for model downloads
- **⚡ Performance**: Caching for repeated queries and optimized operations
- **🏗️ Modular Architecture**: Choose between single-file or modular structure
- **📝 Better Documentation**: Comprehensive setup guides and help

### 🎯 Core Features

- 🔍 **Semantic Search** using FAISS and HuggingFace embeddings (`all-MiniLM-L6-v2`)
- 🌐 **Real-time Web Search** with DuckDuckGo for recent or unknown claims
- 🧠 **Vector Store Memory** to retain and reuse learned evidence
- 🤖 **LLM Verdict Generation** via Groq API using LLaMA 3.1
- 🎨 **Optional Funny Image Generation** using Hugging Face Inference API (black-forest-labs/FLUX.1-dev)
- 🧰 **Enhanced Gradio Interface** for an interactive, user-friendly experience
- 📜 **Comprehensive Logging** for transparency and debugging

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

## 📦 Quick Setup

1. **Clone and Install**
   ```bash
   git clone https://github.com/prshanthreddy/mythbuster.git
   cd mythbuster
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys:
   # GROQ_API_KEY=your_groq_api_key_here
   # HF_API_TOKEN=your_huggingface_api_token_here  # Optional for images
   ```

3. **Run the Application**
   ```bash
   python app.py                # Enhanced single-file version
   # OR
   python app_enhanced.py       # Modular architecture version
   ```

4. **Access the Interface**
   Open your browser to the provided URL (usually http://localhost:7860)

## ✨ Example Myths to Try

- “Drinking cold water causes a sore throat”
- “Humans only use 10% of their brain”
- “Goldfish have a 3-second memory”
- “Lightning never strikes the same place twice”

## 📁 Project Structure

### Single-File Version
```
.
├── app.py                   # Enhanced single-file application
├── app_original.py          # Original backup
├── requirements.txt         # Dependencies
├── .env.template           # Environment template
├── SETUP.md                # Detailed setup guide
└── README.md               # This file
```

### Modular Version  
```
.
├── app_enhanced.py         # Modular entry point
├── config.py               # Configuration management
├── models.py               # Data models and types
├── services.py             # Core business logic
├── ui.py                   # Enhanced Gradio interface
├── utils.py                # Utility functions
├── logger.py               # Logging configuration
└── [common files as above]
```

## 🧠 How It Works

1. **Input Processing** → Sanitize and validate user claims
2. **Memory Search** → Check vector store for similar prior evidence  
3. **Web Research** → If needed, search DuckDuckGo for current information
4. **AI Analysis** → Send context to Groq LLaMA 3.1 for analysis
5. **Verdict Generation** → Return BUSTED ❌ / PLAUSIBLE ❓ / CONFIRMED ✅
6. **Knowledge Storage** → Save new evidence in vector database
6. **Optional Funny Image Generation** → if enabled via checkbox, uses Groq to generate a humorous image prompt, then calls Hugging Face’s `black-forest-labs/FLUX.1-dev` model to generate a matching image
7. **Log + Store Evidence** → saves new info into FAISS vector DB

## 🔐 Security

Make sure not to share your `.env` file or any API keys. Keep your `GROQ_API_KEY` and `HF_API_TOKEN` secure and never commit them to version control.

## 🚀 Deployment Options

### Development
```bash
python app.py              # Quick start with enhanced features
```

### Production  
```bash
python app_enhanced.py     # Modular architecture for production
```

## 🛠️ Troubleshooting

- **"Could not load embedding model"**: Internet required for first run, then cached locally
- **"GROQ_API_KEY not found"**: Copy `.env.template` to `.env` and add your API key  
- **"Image generation disabled"**: Add `HF_API_TOKEN` to `.env` (optional feature)
- **Web search errors**: Rate limiting is normal, responses are cached

## 🤝 Contributing

1. Use the modular structure (`app_enhanced.py`) for new features
2. Follow existing code style and type hints
3. Add tests for new functionality
4. Update documentation

For detailed setup instructions, see [SETUP.md](SETUP.md).
