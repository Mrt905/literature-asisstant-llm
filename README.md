# Literature Assistant LLM

A RAG assistant for tumor microbiome scientific literature.

## Setup

### 1. Start Docker container
First time only:
```bash
docker run -d \
    --name pgvector-papers \
    -e POSTGRES_USER=user \
    -e POSTGRES_PASSWORD=pswd \
    -e POSTGRES_DB=papers \
    -v pgvector_papers:/var/lib/postgresql/data \
    -p 5432:5432 \
    pgvector/pgvector:pg17
```

Every time after:
```bash
docker start pgvector-papers
```

### 2. Install dependencies
```bash
uv add pymupdf psycopg sentence-transformers openai python-dotenv
```

### 3. Ingest papers
- Add PDFs to `papers_pdf/` folder
- Run `ingest.ipynb`

