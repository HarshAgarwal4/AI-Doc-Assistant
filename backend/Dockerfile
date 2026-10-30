FROM python:3.12-slim

# Python environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies without installing the project itself
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code
COPY main.py ./
COPY app/ ./app/
COPY database/ ./database/

# Python module discovery
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Application port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
