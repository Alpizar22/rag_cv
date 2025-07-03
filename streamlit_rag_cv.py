import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# Configuración de la página
st.set_page_config(
    page_title="Asistente de CV RAG",
    page_icon="📄",
    layout="wide"
)

# Variables de entorno
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
    try:
        os.environ["OPENAI_API_KEY"] = api_key

        import tempfile

        # Guardar temporalmente el archivo subido
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        st.success(f"✅ {len(docs)} documento(s) cargado(s)")

        # Dividir documentos
        with st.spinner("Procesando documentos..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            st.success(f"✅ {len(splits)} fragmentos creados")

        # Crear vectorstore usando FAISS (sin SQLite)
        with st.spinner("Creando embeddings..."):
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever()
            st.success("✅ Base vectorial creada con FAISS")

        # Configurar LLM
        with st.spinner("Inicializando modelo de IA..."):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            template = """Eres un analista profesional de CV. Responde la pregunta solamente con el siguiente contexto, poniendo especial atención a fechas, periodos de tiempo e información cronológica:

{context}

Pregunta: {question}

Instrucciones:
- Si la pregunta involucra fechas o periodos de tiempo, sé muy preciso y específico
- Incluye fechas exactas cuando estén disponibles (por ejemplo: "mayo 2024 - presente", "2005-2014")
- Si la información no está en el contexto, responde "No tengo suficiente información"
- Para niveles de experiencia, considera años trabajados, roles y responsabilidades
- Responde de forma estructurada y clara

Respuesta:"""

            prompt = ChatPromptTemplate.from_template(template)

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            st.success("✅ Modelo de IA listo")

        return rag_chain, retriever

    except Exception as e:
        st.error(f"❌ Error cargando el sistema RAG: {str(e)}")
        return None, None


def main():
    st.title("📄 Asistente de CV con RAG")
    api_key = st.secrets["OPENAI_API_KEY"]

    if not api_key:
        st.warning("Por favor configura tu API Key en secrets.toml para usar la app.")
        st.stop()
    
    uploaded_file = st.file_uploader("📄 Sube tu CV (en formato PDF)", type="pdf")

    if not uploaded_file:
        st.warning("Por favor sube un archivo PDF para continuar.")
        st.stop()
    
    rag_chain, retriever = load_rag_system(api_key, uploaded_file)
    st.markdown("Haz preguntas sobre el CV y obtén respuestas con IA:")

    if rag_chain is None:
        st.error("No se pudo cargar el sistema RAG, revisa tu configuración.")
        return

    with st.sidebar:
        st.header("💡 Preguntas de ejemplo")
        for question in [
            "¿Cuál es el nivel del candidato en X habilidad?",
            "¿Cuáles son las fechas exactas de X?",
            "¿Qué estudios tiene el candidato?",
            "¿Cuál es el rol actual del candidato?"
        ]:
            if st.button(question):
                st.session_state.question = question

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🤔 Haz una pregunta")
        question = st.text_input(
            "Pregunta sobre el CV:",
            value=st.session_state.get('question', ''),
            placeholder="Ej: ¿Cuánto tiempo trabajó en X?"
        )
        if st.button("🔍 Obtener respuesta"):
            if question.strip():
                with st.spinner("🤖 Pensando..."):
                    try:
                        answer = rag_chain.invoke(question)
                        st.subheader("💡 Respuesta")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"❌ Error al obtener respuesta: {str(e)}")
            else:
                st.warning("⚠️ Por favor escribe una pregunta.")
    
    with col2:
        st.header("ℹ️ Acerca de")
        st.markdown("""
        Este sistema RAG analiza el CV y responde preguntas con IA.
        - Procesa PDFs
        - Usa búsqueda semántica
        - Preciso en fechas
        """)

if __name__ == "__main__":
    main()
