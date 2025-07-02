# CV RAG Assistant - Streamlit App

A simple Streamlit application that uses RAG (Retrieval-Augmented Generation) to answer questions about Leonardo's CV.

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

2. Make sure your CV PDF file is in the correct path:
   - Update the file path in `streamlit_rag_cv.py` if needed
   - Current path: `C:\Users\leo\Documents\CV\Leonardo Ferreira da Silva CV - English.pdf`

3. Ensure your API keys are set in the environment variables (already configured in the code)

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_rag_cv.py
```

The app will:
1. Load and process your CV documents
2. Create vector embeddings
3. Provide a web interface for asking questions
4. Display AI-generated answers with relevant CV sections

## Example Questions

- "What is his Python level?"
- "What level is the person? Junior, semi or senior?"
- "What are the exact dates of their PhD program?"
- "How long have they been working at Outlier?"
- "What is their current role and when did they start?"

## Architecture

- **Document Loading**: PyPDFLoader for PDF processing
- **Text Splitting**: RecursiveCharacterTextSplitter with 1000 char chunks
- **Vector Store**: Chroma with OpenAI embeddings
- **LLM**: GPT-3.5-turbo via OpenAI
- **UI**: Streamlit with responsive layout

## Improvements Made

- Enhanced prompt for better date handling
- Larger chunk sizes (1000 chars) to preserve temporal context
- Caching with `@st.cache_resource` for better performance
- Error handling and user feedback
- Expandable sections to view relevant CV content 