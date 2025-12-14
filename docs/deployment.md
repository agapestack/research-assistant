# Deployment

How to deploy the Research Assistant in various environments.

## Local Development

### Prerequisites

- Python 3.12+
- Docker (for Qdrant)
- Node.js 18+ (for frontend)
- Ollama (for LLM)

### Quick Start

```bash
# 1. Clone and install
git clone https://github.com/agapestack/research-assistant
cd research-assistant
uv sync

# 2. Start Qdrant
docker compose up -d qdrant

# 3. Pull an LLM model
ollama pull qwen3:14b

# 4. Index some papers
uv run python scripts/benchmark_embeddings.py --models bge-base --papers 20

# 5. Start the API
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 6. Start the frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Environment Variables

Create a `.env` file:

```bash
# Qdrant
RAG_QDRANT_HOST=localhost
RAG_QDRANT_PORT=6333

# Embeddings
RAG_EMBEDDING_MODEL=bge-base

# LLM
RAG_LLM_MODEL=qwen3:14b
RAG_LLM_TEMPERATURE=0.1

# Retrieval
RAG_RETRIEVAL_K=5
```

---

## Docker Compose (Full Stack)

### docker-compose.yml

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RAG_QDRANT_HOST=qdrant
      - RAG_EMBEDDING_MODEL=bge-base
    depends_on:
      - qdrant

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - api

volumes:
  qdrant_data:
```

### Dockerfile (API)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy and install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

# Copy source code
COPY src/ ./src/

# Run the API
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
docker compose up -d
```

---

## Production Considerations

### 1. Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### 2. SSL with Certbot

```bash
sudo certbot --nginx -d api.yourdomain.com
```

### 3. Process Manager (systemd)

```ini
# /etc/systemd/system/research-assistant.service
[Unit]
Description=Research Assistant API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/research-assistant
ExecStart=/home/ubuntu/.local/bin/uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable research-assistant
sudo systemctl start research-assistant
```

### 4. Environment Security

```bash
# Never commit .env files
echo ".env" >> .gitignore

# Use secrets manager in production
# AWS: Secrets Manager
# GCP: Secret Manager
# Azure: Key Vault
```

---

## Cloud Deployment

### AWS (EC2 + Docker)

```bash
# 1. Launch EC2 instance (t3.medium or larger)
# 2. Install Docker
sudo yum install docker -y
sudo systemctl start docker

# 3. Install docker-compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 4. Clone and run
git clone https://github.com/agapestack/research-assistant
cd research-assistant
docker-compose up -d
```

### GCP (Cloud Run)

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/research-assistant

# Deploy
gcloud run deploy research-assistant \
  --image gcr.io/PROJECT_ID/research-assistant \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Railway / Render

Both support direct GitHub deployment:

1. Connect your repository
2. Set environment variables
3. Deploy automatically on push

---

## Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Qdrant health
curl http://localhost:6333/health
```

### Logging

Configure structured logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Metrics (Future)

Consider adding:

- Prometheus metrics endpoint
- Request latency histograms
- Error rate tracking
- Vector store size monitoring

---

## Scaling

### Horizontal Scaling

```
                    ┌─────────────┐
                    │ Load        │
                    │ Balancer    │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │   API #1    │ │   API #2    │ │   API #3    │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           └───────────────┼───────────────┘
                    ┌──────▼──────┐
                    │   Qdrant    │
                    │  (Cluster)  │
                    └─────────────┘
```

### Qdrant Cluster

For high availability:

```yaml
# docker-compose.yml
services:
  qdrant-node-1:
    image: qdrant/qdrant:latest
    environment:
      - QDRANT__CLUSTER__ENABLED=true
    # ...

  qdrant-node-2:
    image: qdrant/qdrant:latest
    environment:
      - QDRANT__CLUSTER__ENABLED=true
    # ...
```

---

## GitHub Pages (Documentation)

This documentation site is deployed via GitHub Actions:

```yaml
# .github/workflows/docs.yml
name: Deploy Documentation

on:
  push:
    branches: [main]
    paths: ['docs/**', 'mkdocs.yml']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install mkdocs-material mkdocs-mermaid2-plugin
      - run: mkdocs gh-deploy --force
```

Access at: `https://agapestack.github.io/research-assistant`
