FROM python:3.11-slim

# Create a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user pyproject.toml .
# Install uv and dependencies
RUN pip install --no-cache-dir uv
RUN uv pip install --system -r pyproject.toml

# Copy the rest of the application
COPY --chown=user . /app

# Expose the port Hugging Face expects
EXPOSE 7860

# We set CHROMA_PERSIST_DIR to /data (the persistent volume provided by Hugging Face)
ENV CHROMA_PERSIST_DIR="/data/chroma_db_groq_legal"

# Start the FastAPI server on port 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
