# Tumor Microbiome Literature Assistant

A RAG (Retrieval-Augmented Generation) assistant that answers questions based on scientific literature about tumor microbiome. The system uses vector search to find relevant chunks from uploaded papers and generates accurate, source-grounded answers using an LLM.

## How it works

1. PDF papers are loaded, chunked and embedded using `pritamdeka/S-PubMedBert-MS-MARCO` — a biomedical embedding model trained on PubMed literature
2. Embeddings are stored in PostgreSQL with pgvector extension
3. At query time, the most similar chunks are retrieved and passed to the LLM as context
4. The LLM generates an answer grounded in the provided literature
5. Every conversation is logged to a monitoring database and visualized in Grafana

## Project structure

literature-assistant-llm/
├── app.py ← Flask web application
├── ragbase.py ← RAG pipeline classes
├── ingest.py ← PDF loading, chunking, embedding functions
├── evaluation.py ← Retrieval evaluation metrics
├── templates/
│ └── index.html ← Web interface
├── grafana/ ← Grafana provisioning
├── papers_pdf/ ← Put your PDF papers here!!!
├── data/ ← Ground truth and evaluation results
├── docker-compose.yml ← Docker services configuration
├── Dockerfile ← Flask app container
├── init.sql ← Database initialization
├── 01_ingest.ipynb ← Ingestion pipeline
├── 02_rag.ipynb ← RAG pipeline notebook
├── 03_evaluation-ret.ipynb ← Ground truth generation
├── 04_evaluation-ret-exp.ipynb ← Embedding model experiments
└── 05_evaluation-rag.ipynb ← RAG answer quality evaluation

## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — for running the ingestion notebook
- OpenAI API key

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Mrt905/literature-assistant-llm.git
cd literature-assistant-llm
```

### 2. Set up environment variables
Create a `.env` file in the root folder:
```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. Start all services
```bash
docker compose up --build
```

This starts:
- **PostgreSQL + pgvector** — vector database for papers at port `5432`
- **Flask app** — RAG assistant at `http://localhost:5000`
- **Grafana** — monitoring dashboard at `http://localhost:3000` (login: admin/admin)

### 4. Ingest papers
Install dependencies for the ingestion notebook:
```bash
uv sync
```

Then:
- Name your PDF files as: `keyword1_keyword2-author-journal-year.pdf`
- Place them in the `papers_pdf/` folder
- Run `01_ingest.ipynb` to load, chunk, embed and store papers in the database

### 5. Open the app
Go to `http://localhost:5000` in your browser.

### 6. Stop services
```bash
docker compose down
```

## Running the app after first setup
```bash
docker compose up
```

## Monitoring
Grafana dashboard at `http://localhost:3000` shows:
- Total questions asked
- Average response time
- Average token usage
- Questions and response time over time
- Recent conversations
- User feedback 

## Evaluation

> Results are specific to the 12 papers used in this project on tumor microbiome. Results will vary depending on the number, topic, and quality of papers provided.

**Best embedding model:** `pritamdeka/S-PubMedBert-MS-MARCO`
- Hit Rate@10: 0.604
- MRR@10: 0.380

**RAG answer quality (evaluated on 200 sample questions):**
- Relevant: 82.6%
- Partly relevant: 17.0%
- Non-relevant: 0.5%

The relatively low retrieval scores are due to all papers being on the same topic (tumor microbiome), making chunks semantically similar and harder to distinguish. 