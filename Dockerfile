FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better caching
COPY unison-inference/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir redis python-jose[cryptography] bleach httpx pyyaml

# Copy service source and shared library from monorepo
COPY unison-inference/src ./src/
COPY unison-common/src/unison_common ./src/unison_common

ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8087

# Run the service
CMD ["python", "src/server.py"]
