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
    """Carga del sistema RAG (no cacheado, depende de la API key)."""
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

        # Crear vectorstore
        with st.spinner("Creando embeddings..."):
            vectorstore = Chroma.from_documents(
            	documents=splits,
            	embedding=OpenAIEmbeddings(),
            	persist_directory=None  # esto fuerza modo in-memory y evita SQLite)
            retriever = vectorstore.as_retriever()
            st.success("✅ Base vectorial creada")

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
    """Aplicación principal Streamlit"""
    
    # Encabezado
    st.title("📄 Asistente de CV con RAG")
    api_key = st.secrets["OPENAI_API_KEY"]

    if not api_key:
        st.warning("Por favor ingresa tu API Key para usar la aplicación.")
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

    # Barra lateral con preguntas ejemplo
    with st.sidebar:
        st.header("💡 Preguntas de ejemplo")
        example_questions = [
            "¿Cuál es el nivel del candidato en X habilidad?",
            "¿Cuál es el nivel de seniority del candidato?",
            "¿Cuáles son las fechas exactas de X?",
            "¿Cuánto tiempo lleva trabajando en X?",
            "¿Cuál es el rol actual del candidato?",
            "¿Cuántos años totales de experiencia tiene?",
            "¿Cuáles son las principales habilidades del candidato?",
            "¿Qué estudios tiene el candidato?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"btn_{question[:30]}"):
                st.session_state.question = question

    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🤔 Haz una pregunta")
        
        question = st.text_input(
            "Escribe tu pregunta sobre el CV:",
            value=st.session_state.get('question', ''),
            placeholder="Ej: ¿Cuál es su nivel en X?"
        )
        
        if st.button("🔍 Obtener respuesta", type="primary"):
            if question.strip():
                with st.spinner("🤖 Pensando..."):
                    try:
                        answer = rag_chain.invoke(question)
                        st.subheader("💡 Respuesta")
                        st.write(answer)
                        
                        with st.expander("📄 Secciones relevantes del CV"):
                            relevant_docs = retriever.get_relevant_documents(question)
                            for i, doc in enumerate(relevant_docs):
                                st.markdown(f"**Sección {i+1}:**")
                                st.text(doc.page_content)
                                st.divider()
                    except Exception as e:
                        st.error(f"❌ Error al obtener respuesta: {str(e)}")
            else:
                st.warning("⚠️ Por favor escribe una pregunta.")
    
    with col2:
        st.header("ℹ️ Acerca de")
        st.markdown("""
        Este sistema RAG (Generación aumentada con recuperación) analiza el CV que cargues para responder tus preguntas.
        
        **Características:**
        - 📄 Procesamiento de documentos PDF
        - 🔍 Búsqueda semántica
        - 🤖 Respuestas con IA
        - 📅 Precisión en fechas
        
        **Cómo funciona:**
        1. Sube y procesa el CV
        2. Crea embeddings semánticos
        3. Busca la información relevante
        4. Genera respuestas contextuales
        """)
        
        st.header("🔧 Estado del sistema")
        if rag_chain:
            st.success("✅ Sistema RAG: Listo")
            st.success("✅ Modelo de IA: Conectado")
            st.success("✅ Documentos: Cargados")
        else:
            st.error("❌ Sistema no listo")


if __name__ == "__main__":
    main()
