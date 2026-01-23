# -----------------------------
# Base image
# -----------------------------
FROM python:3.11-slim

# -----------------------------
# Environment settings
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# System dependencies (minimal)
# -----------------------------
# Added curl for healthchecks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Python dependencies
# -----------------------------
COPY requirements.txt .

# CRITICAL FIX: Install CPU-only PyTorch BEFORE requirements.
# This prevents pip from downloading 5GB+ of Nvidia CUDA libraries.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy application code
# -----------------------------
COPY . .

# -----------------------------
# Expose Streamlit port
# -----------------------------
EXPOSE 8501

# -----------------------------
# Healthcheck (Ensures app is actually ready)
# -----------------------------
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# -----------------------------
# Streamlit entrypoint
# -----------------------------
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]