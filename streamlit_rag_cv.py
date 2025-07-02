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

os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system components."""
    try:
        # Load documents
        with st.spinner("Loading CV documents..."):
            loader = PyPDFLoader(
                r'C:\Users\leo\Documents\CV\Leonardo Ferreira da Silva CV - English.pdf'
            )
            docs = loader.load()
            st.success(f"✅ Loaded {len(docs)} document(s)")

        # Split documents
        with st.spinner("Processing documents..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)
            st.success(f"✅ Created {len(splits)} text chunks")

        # Create vector store
        with st.spinner("Creating vector embeddings..."):
            vectorstore = Chroma.from_documents(
                documents=splits, embedding=OpenAIEmbeddings()
            )
            retriever = vectorstore.as_retriever()
            st.success("✅ Vector store created")

        # Setup LLM and chain
        with st.spinner("Initializing AI model..."):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            
            def format_docs(docs):
                """Format documents for the prompt."""
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
    st.markdown("Ask questions about Leonardo's CV and get AI-powered answers!")
    
    # Load RAG system
    rag_chain, retriever = load_rag_system()
    
    if rag_chain is None:
        st.error("Failed to load the RAG system. Please check your configuration.")
        return
    
    # Sidebar with example questions
    with st.sidebar:
        st.header("💡 Example Questions")
        example_questions = [
            "What is his Python level?",
            "What level is the person? Junior, semi or senior?",
            "What are the exact dates of their PhD program?",
            "How long have they been working at Outlier?",
            "What is their current role and when did they start?",
            "What is their total years of experience?",
            "What are their main skills?",
            "What education does he have?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"btn_{question[:20]}"):
                st.session_state.question = question
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🤔 Ask a Question")
        
        # Text input for question
        question = st.text_input(
            "Enter your question about the CV:",
            value=st.session_state.get('question', ''),
            placeholder="e.g., What is his Python level?"
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
        This RAG (Retrieval-Augmented Generation) system analyzes Leonardo's CV to answer your questions.
        
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