FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application package (tests and docs stay out of the image)
COPY app/ ./app/

EXPOSE 8000

# Render injects $PORT at runtime; fall back to 8000 for local Docker runs
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
