FROM python:3.10-slim

# Create a non-root user and group
RUN groupadd -r traceiq && useradd -r -g traceiq -s /sbin/nologin -c "Docker image user" traceiq

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY apps/ ./apps/
COPY configs/ ./configs/
COPY data/ ./data/

# Change ownership to the non-root user
RUN chown -R traceiq:traceiq /app

# Switch to the non-root user
USER traceiq

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
