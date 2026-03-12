FROM debian:bookworm-slim AS frontend-builder
WORKDIR /app

# Install curl to download the standalone binary
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the frontend files
COPY index.html .
COPY tailwind.config.js .
COPY input.css .

# Download the Tailwind Standalone CLI, make it executable, and compile the CSS!
RUN curl -sLO https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-linux-x64 \
    && chmod +x tailwindcss-linux-x64 \
    && ./tailwindcss-linux-x64 -i ./input.css -o ./static/output.css --minify

# ==========================================
# STAGE 2: Python Production Runtime
# ==========================================
FROM python:3.11-slim
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Copy the Python backend and HTML
COPY app.py .
COPY index.html .
COPY favicon.ico .

# COPY THE COMPILED CSS FROM STAGE 1 (Leaves Node.js behind!)
COPY --from=frontend-builder /app/static ./static

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]