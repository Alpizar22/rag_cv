# Docker image name
IMAGE_NAME=rag_cv

# Streamlit port
PORT=8501

# Build image from Dockerfile
build:
	docker build -t $(IMAGE_NAME) .

# Build directly from GitHub
build-remote:
	docker build -t $(IMAGE_NAME) https://github.com/leofds12/rag_cv.git

# Run container
run:
	docker run -p $(PORT):8501 $(IMAGE_NAME)

# Clean containers and images
clean:
	docker container prune -f
	docker image prune -f

# Build + run in one step
start: build run
