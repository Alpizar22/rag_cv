FROM python:3.12-slim

# Evita prompts innecesarios
ENV DEBIAN_FRONTEND=noninteractive

# Define el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el contenido del repositorio al contenedor
COPY . .

# Instala dependencias del sistema si alguna lo necesita (puede omitirse si no usás Chroma localmente)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Instala las dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone el puerto usado por Streamlit
EXPOSE 8501

# Comando que ejecuta la app cuando arranca el contenedor
CMD ["streamlit", "run", "streamlit_rag_cv.py"]
