# CV RAG Assistant - Streamlit App

A Streamlit application that uses LangChain and Retrieval-Augmented Generation (RAG) to answer questions about any uploaded CV using OpenAI's GPT models.


## Features

- 📄 PDF document processing
- 🔍 Semantic search with vector embeddings
- 🤖 AI-powered responses using GPT-3.5-turbo
- 📅 Date-aware answers
- 💡 Example questions sidebar
- 📊 System status monitoring

## Installation

1. Clone the repository:
```bash
git clone https://github.com/leofds12/rag_cv.git
cd rag_cv
```

2. Create and activate a virtual environment (optional but recommended):

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app
```bash
streamlit run streamlit_rag_cv.py
```

5. Enter your personal OpenAI API Key in the input field to use the app (do not share it!) If you don't have one, create a new one in https://openai.com/api/ (you must load some credit, 1 dollar is more than enough for many requests with GPT-3.5-turbo model).

5. In the web app, upload your CV in PDF format.


The app will:
- Load and process your CV documents
- Create vector embeddings
- Provide a web interface for asking questions
- Display AI-generated answers with relevant CV sections

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
- **LangChain**: Core framework to orchestrate document retrieval + LLM
- **Document Loading**: PyPDFLoader for PDF processing
- **Text Splitting**: RecursiveCharacterTextSplitter
- **Vector Store**: Chroma with OpenAI embeddings
- **LLM**: GPT-3.5-turbo via OpenAI
- **UI**: Streamlit with responsive layout

🛠 Makefile Support (Optional)
A simple Makefile is included for convenience. Run these commands from the project root:

```bash
make install     # Install Python dependencies
make run         # Launch the Streamlit app
make format      # Auto-format Python code with black
make lint        # Run flake8 to check style issues
```


🐳 Docker Support (Optional)
You can also build and run the app using Docker:

1. Build the Docker image:
```bash
docker build -t rag_cv .
```

```bash
docker run -p 8501:8501 -v $(pwd):/app rag_cv
```

This app does not store any documents or data — everything runs locally unless your API key usage is tracked by OpenAI.

Your OpenAI API key is not logged or stored by the app.

The system is designed to work with any CV in English and may behave differently for other document types.