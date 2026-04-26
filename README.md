# Workplace Assistant Chatbot

A chatbot application for workplace assistance using GROQ API for LLM functionality, with support for:
- JSON-based employee data queries
- Policy document search using RAG (Retrieval Augmented Generation)
- Vector database for efficient document retrieval

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/tanayagarwal/Documents/Workplace-Assistant-Chatbot
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Then edit `.env` and add your API keys:
```
GROQ_API_KEY=your_actual_groq_api_key_here
OPENWEATHER_API_KEY=your_actual_openweather_api_key_here
GROQ_MODEL=mixtral-8x7b-32768
```

### 3. Initialize the Database (First Time Only)

```bash
python create_database.py --reset
```

This will:
- Load PDF documents from the `pdf_data` directory
- Create embeddings using HuggingFace
- Store them in a Chroma vector database

## 📖 Usage

### Combined Chatbot

Run the main chatbot application:

```bash
python combined_chatbot.py
```

This demonstrates both JSON queries and policy document searches.

### JSON Query App

Query employee data from JSON:

```bash
python json_app.py "who is my manager?"
python json_app.py "how many leave days do I have remaining?"
```

### Policy Query App

Search policy documents:

```bash
python policy_app.py "what is the marriage gift policy?"
python policy_app.py "how many days of privilege leave can I take?"
```

## 🔧 Configuration

### GROQ Models

Available models (configure in `.env`):
- `mixtral-8x7b-32768` (default) - Good balance of speed and quality
- `llama2-70b-4096` - Higher quality, slower
- `gemma-7b-it` - Faster, smaller model
- `llama3-70b-8192` - Latest model with larger context

### Embeddings

Currently using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` for embeddings. This runs locally and doesn't require any additional API keys.

## 📁 Project Structure

```
├── combined_chatbot.py      # Main chatbot application (converted from notebook)
├── create_database.py        # Vector database initialization
├── json_app.py              # JSON query application
├── policy_app.py            # Policy document search application
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .env                     # Your actual API keys (gitignored)
├── leavesEmployee.json      # Employee data
├── pdf_data/                # Policy documents (PDFs)
└── chroma/                  # Vector database storage
```

## 🔄 Migration from Ollama/Azure OpenAI

This project has been migrated from using Ollama and Azure OpenAI to GROQ API:

### Changes Made:
- ✅ Replaced `Ollama` LLM with `ChatGroq`
- ✅ Replaced `OllamaEmbeddings` with `HuggingFaceEmbeddings`
- ✅ Fixed deprecated imports:
  - `langchain.text_splitter` → `langchain_text_splitters`
  - `langchain.prompts` → `langchain_core.prompts`
- ✅ Converted Jupyter notebook to Python script
- ✅ Added environment variable support with `python-dotenv`

## 🐛 Troubleshooting

### "Module not found" errors
Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### "API key not found" errors
Ensure your `.env` file exists and contains valid API keys.

### Slow embedding generation (first run)
The first time you run the code, HuggingFace will download the embedding model (~80MB). Subsequent runs will be faster.

### Vector database errors
If you encounter database errors, try resetting it:
```bash
python create_database.py --reset
```

## 📝 Notes

- The original Jupyter notebook `combined-chatbot.ipynb` is preserved for reference
- All imports have been updated to use the latest langchain packages
- GROQ API provides faster inference compared to local Ollama models
- HuggingFace embeddings run locally with no API costs
