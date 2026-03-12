FROM python:3.11-slim

# Git is required for some Hugging Face hub operations
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force Hugging Face to use the Rust-based hf_transfer library globally
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Copy your app.py containing the FastAPI backend and Web UI
COPY app.py .

# Expose the port for the Web UI
EXPOSE 8000

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]