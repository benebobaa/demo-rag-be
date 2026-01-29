FROM python:3.13-slim-bookworm AS builder

# Install uv
RUN pip install uv

# Copy project definition
WORKDIR /app
COPY pyproject.toml uv.lock ./

# Sync dependencies to create .venv
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.13-slim-bookworm

# Copy virtual env from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . /app
WORKDIR /app

# Set environment to use the venv
ENV PATH="/app/.venv/bin:$PATH"

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
