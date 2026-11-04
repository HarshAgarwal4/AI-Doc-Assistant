
# Python base image
FROM python:3.12-slim

# Python environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install production dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code
COPY main.py ./
COPY app/ ./app/
COPY database/ ./database/
COPY utils/ ./utils/

# Copy Alembic configuration and migrations
COPY migrations/ ./migrations/
COPY alembic.ini ./

# Python module discovery and virtual environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

# FastAPI port
EXPOSE 5000

# Start FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
