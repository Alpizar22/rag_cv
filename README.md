# CV RAG Assistant - Streamlit App

A Streamlit application that uses RAG (Retrieval-Augmented Generation) to answer questions about any uploaded CV.

## Features

- 📄 PDF document processing
- 🔍 Semantic search with vector embeddings
- 🤖 AI-powered responses using GPT-3.5-turbo
- 📅 Date-aware answers
- 💡 Example questions sidebar
- 📊 System status monitoring

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app

3. Enter your personal OpenAI API Key in the input field to use the app (sharing your private key is not recommended).

4. In the web app, upload your CV PDF file (no specific filename required).


The app will:
1. Load and process your CV documents
2. Create vector embeddings
3. Provide a web interface for asking questions
4. Display AI-generated answers with relevant CV sections

## Example Questions

   "What is the candidate level on X skill?",
   "What is the candidate seniority?",
   "What are the exact dates of X?",
   "How long has the candidate working at X?",
   "What is the candidate current role?",
   "How many total years of experience?",
   "What are the candidate main skills?",
   "What education does the candidate have?"

## Architecture

- **Document Loading**: PyPDFLoader for PDF processing
- **Vector Store**: Chroma with OpenAI embeddings
- **LLM**: GPT-3.5-turbo via OpenAI
- **UI**: Streamlit with responsive layout