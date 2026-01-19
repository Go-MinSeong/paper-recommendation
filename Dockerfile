# ============================================
# Base stage - common dependencies
# ============================================
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ============================================
# Development stage
# ============================================
FROM base AS development

# Copy dependency files
COPY pyproject.toml README.md ./

# Install all dependencies including dev
RUN pip install --no-cache-dir -e ".[dev]"

# Create necessary directories
RUN mkdir -p data logs

EXPOSE 8000

CMD ["python", "main.py"]

# ============================================
# Production stage
# ============================================
FROM base AS production

# Copy dependency files
COPY pyproject.toml README.md ./

# Install production dependencies only
RUN pip install --no-cache-dir .

# Copy application code
COPY config/ ./config/
COPY mcp_servers/ ./mcp_servers/
COPY src/ ./src/
COPY main.py ./

# Create necessary directories
RUN mkdir -p data logs

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py"]
