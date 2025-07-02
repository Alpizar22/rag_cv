import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# Page configuration
st.set_page_config(
    page_title="CV RAG Assistant",
    page_icon="📄",
    layout="wide"
)

# Environment variables
load_dotenv()

tracing = os.getenv("LANGCHAIN_TRACING_V2")
endpoint = os.getenv("LANGCHAIN_ENDPOINT")
api_key = os.getenv("LANGCHAIN_API_KEY")

if tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = tracing
if endpoint:
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint
if api_key:
    os.environ["LANGCHAIN_API_KEY"] = api_key


def load_rag_system(api_key, uploaded_file):
    """Load the RAG system components (not cached, depends on API key)."""
    try:
        #os.environ["OPENAI_API_KEY"] = api_key  # set key before using any LangChain component

        import tempfile

        # Guardar temporalmente el archivo subido
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        st.success(f"✅ Loaded {len(docs)} document(s)")

        # Split documents
        with st.spinner("Processing documents..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            st.success(f"✅ Created {len(splits)} text chunks")

        # Create vector store
        with st.spinner("Creating vector embeddings..."):
            vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
            retriever = vectorstore.as_retriever()
            st.success("✅ Vector store created")

        # Setup LLM and chain
        with st.spinner("Initializing AI model..."):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # Enhanced prompt for better date handling
            template = """You are a professional CV analyzer. Answer the question based only on the following context, paying special attention to dates, time periods, and chronological information:

{context}

Question: {question}

Instructions:
- If the question involves dates or time periods, be very precise and specific
- Include exact dates when available (e.g., "May 2024 - Present", "2005-2014")
- If information is not in the context, say "I don't have enough information"
- For experience levels, consider years of experience, roles, and responsibilities
- Provide structured, clear responses

Answer:"""

            prompt = ChatPromptTemplate.from_template(template)

            # Create RAG chain
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            st.success("✅ AI model ready")

        return rag_chain, retriever

    except Exception as e:
        st.error(f"❌ Error loading RAG system: {str(e)}")
        return None, None


def main():
    """Main Streamlit application."""
    
 
    # Header
    st.title("📄 CV RAG Assistant")
    api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")


    if not api_key:
        st.warning("Please enter your API key to use the app.")
        st.stop()
    
    uploaded_file = st.file_uploader("📄 Upload your CV (PDF format)", type="pdf")

    if not uploaded_file:
        st.warning("Please upload a PDF file to continue.")
        st.stop()    
        
    rag_chain, retriever = load_rag_system(api_key, uploaded_file)
    #os.environ["OPENAI_API_KEY"] = api_key
    st.markdown("Ask questions about the CV and get AI-powered answers!")
    
    if rag_chain is None:
        st.error("Failed to load the RAG system. Please check your configuration.")
        return
    
    # Sidebar with example questions
    with st.sidebar:
        st.header("💡 Example Questions")
        example_questions = [
            "What is the candidate level on X skill?",
            "What is the candidate seniority?",
            "What are the exact dates of X?",
            "How long has the candidate working at X?",
            "What is the candidate current role?",
            "How many total years of experience?",
            "What are the candidate main skills?",
            "What education does the candidate have?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"btn_{question[:30]}"):
                st.session_state.question = question
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🤔 Ask a Question")
        
        # Text input for question
        question = st.text_input(
            "Enter your question about the CV:",
            value=st.session_state.get('question', ''),
            placeholder="e.g., What is his level at X?"
        )
        
        # Submit button
        if st.button("🔍 Get Answer", type="primary"):
            if question.strip():
                with st.spinner("🤖 Thinking..."):
                    try:
                        # Get answer
                        answer = rag_chain.invoke(question)
                        
                        # Display answer
                        st.subheader("💡 Answer")
                        st.write(answer)
                        
                        # Show relevant documents
                        with st.expander("📄 View Relevant CV Sections"):
                            relevant_docs = retriever.get_relevant_documents(question)
                            for i, doc in enumerate(relevant_docs):
                                st.markdown(f"**Section {i+1}:**")
                                st.text(doc.page_content)
                                st.divider()
                                
                    except Exception as e:
                        st.error(f"❌ Error getting answer: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question.")
    
    with col2:
        st.header("ℹ️ About")
        st.markdown("""
        This RAG (Retrieval-Augmented Generation) system analyzes the uploaded CV to answer your questions.
        
        **Features:**
        - 📄 PDF document processing
        - 🔍 Semantic search
        - 🤖 AI-powered responses
        - 📅 Date-aware answers
        
        **How it works:**
        1. Uploads and processes CV documents
        2. Creates vector embeddings
        3. Searches for relevant information
        4. Generates contextual answers
        """)
        
        # System status
        st.header("🔧 System Status")
        if rag_chain:
            st.success("✅ RAG System: Ready")
            st.success("✅ AI Model: Connected")
            st.success("✅ Documents: Loaded")
        else:
            st.error("❌ System: Not Ready")


if __name__ == "__main__":
    main() 