# MythBuster AI - Setup Guide

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/prshanthreddy/mythbuster.git
   cd mythbuster
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

## Enhanced Features

### ✨ What's New
- **Better Error Handling**: Graceful fallbacks when services are unavailable
- **Input Validation**: Sanitization and validation of user inputs
- **Improved UI**: Better styling, loading indicators, and user feedback
- **Offline Resilience**: Works even when some services are offline
- **Enhanced Logging**: Better monitoring and debugging information
- **Type Safety**: Comprehensive type hints for better code quality

### 🏗️ Architecture Options

#### Option 1: Single File (app.py)
- **Pros**: Simple to deploy, easy to understand
- **Cons**: Less maintainable for large changes
- **Best for**: Quick deployments, simple setups

#### Option 2: Modular Structure (app_enhanced.py)
- **Pros**: Better organization, easier to maintain and extend
- **Cons**: Slightly more complex setup
- **Best for**: Development, production environments

### 🔧 Configuration

The application supports flexible configuration through environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| GROQ_API_KEY | Yes | API key for LLaMA 3 analysis |
| HF_API_TOKEN | No | Token for image generation (optional) |

### 🚀 Deployment Options

#### Local Development
```bash
python app.py
```

#### Production with Enhanced Structure
```bash
python app_enhanced.py
```

#### Docker (Future Enhancement)
```bash
# Coming soon...
docker build -t mythbuster .
docker run -p 7860:7860 mythbuster
```

### 🛠️ Troubleshooting

#### Common Issues

1. **"Could not load embedding model"**
   - Ensure internet connection for first run
   - Model will be cached locally after first download

2. **"GROQ_API_KEY not found"**
   - Create `.env` file from `.env.template`
   - Add your API key from https://console.groq.com/keys

3. **"Image generation disabled"**
   - Add HF_API_TOKEN to `.env` file
   - Get token from https://huggingface.co/settings/tokens

4. **Web search errors**
   - Rate limiting from DuckDuckGo is normal
   - Responses cached to reduce repeated calls

### 📊 Performance Tips

1. **First Run**: May take longer due to model downloads
2. **Caching**: Repeated queries are faster due to vector storage
3. **Memory**: Vector store grows over time, improving accuracy
4. **Network**: Works offline after initial model download

### 🔐 Security Considerations

1. **API Keys**: Never commit `.env` files to version control
2. **Input Validation**: All user inputs are sanitized
3. **Rate Limiting**: Built-in caching reduces API calls
4. **Logging**: No sensitive data logged

### 🤝 Contributing

To contribute to the enhanced version:

1. Use the modular structure in `app_enhanced.py`
2. Add tests for new features
3. Follow the existing code style
4. Update documentation for new features