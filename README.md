# Tumor Microbiome Literature Assistant

Try asking questions about tumor microbiome here:

[Tumor Microbiome Assistant](https://literature-rag-production.up.railway.app/)

A RAG (Retrieval-Augmented Generation) system designed for **scientific literature research**. The system allows researchers to quickly retrieve relevant information from a curated collection of scientific papers using natural language questions.

> **Important:** This tool is designed to assist researchers, not replace careful reading of the literature. The recommended workflow is:
> 1. Collect and curate relevant papers on your topic
> 2. Ingest them into the system
> 3. Use the assistant for fast retrieval and initial orientation
> 4. Always read the original papers before drawing conclusions

The system was built and tested using papers on **tumor microbiome** as a use case, but can be used for any scientific literature by providing your own PDF papers. The open source papers used are referenced in references.md.

## How it works

1. PDF papers are loaded, chunked and embedded using `pritamdeka/S-PubMedBert-MS-MARCO` — a biomedical embedding model trained on PubMed literature
2. Embeddings are stored in PostgreSQL with pgvector extension
3. At query time, hybrid search (vector + text) retrieves the top 20 most relevant chunks, which are then reranked by a cross-encoder
4. The top 10 chunks are passed to the LLM (`gpt-4o-mini`) as context
5. The LLM generates an answer grounded in the provided literature
6. Every conversation is logged to a monitoring database and visualized in Grafana


## Project structure

```
literature-assistant-llm/
├── app/
│   ├── app.py                              ← Flask web application
│   ├── ragbase.py                          ← RAG pipeline classes
│   └── templates/
│       └── index.html                      ← Web interface
├── pipelines/
│   ├── ingest.py                           ← PDF loading, chunking, embedding functions
│   └── evaluation.py                       ← Retrieval evaluation metrics
├── notebooks/
│   ├── 03_ground_truth_generation.ipynb    ← Ground truth generation
│   ├── 04_evaluation-ret-exp.ipynb         ← Embedding model experiments
│   ├── 05_evaluation-ret-exp2.ipynb        ← Search strategy experiments
│   └── 06_evaluation-rag.ipynb             ← RAG answer quality evaluation
├── data/
│   └── raw/
│       └── papers_pdf/                     ← Put your PDF papers here
├── grafana/                                ← Grafana provisioning
├── 01_ingest.ipynb                         ← Ingestion pipeline (run when adding papers)
├── 02_rag.ipynb                            ← RAG pipeline notebook
├── docker-compose.yml                      ← Docker services configuration
├── Dockerfile                              ← Flask app container
├── init.sql                                ← Database initialization
├── railway.json                            ← Railway deployment configuration
├── REFERENCES.md                           ← List of papers used
└── README.md
```


## What was implemented (project criteria)

| Criteria | Notes |
|----------|-------|
| **Problem description** | RAG assistant for scientific literature research |
| **Retrieval flow** | Knowledge base (PostgreSQL + pgvector) and LLM (GPT) used in the flow |
| **Retrieval evaluation** | 4 embedding models and 5 search strategies evaluated, best one selected |
| **LLM evaluation** | 2 LLM models and 2 prompt templates evaluated, best one selected |
| **Interface** | Flask web application with chat interface |
| **Ingestion pipeline** | Semi-automated ingestion via `01_ingest.ipynb` |
| **Monitoring** | User feedback (👍/👎) collected + Grafana dashboard with 5+ charts |
| **Containerization** | Everything in Docker Compose (app, PostgreSQL, Grafana) |
| **Reproducibility** | Clear instructions in README, papers included, dependencies specified in `pyproject.toml` |
| **Hybrid search** | Hybrid search (vector + BM25) implemented and evaluated |
| **Document re-ranking** | Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) implemented and evaluated |
| **User query rewriting** | Query rewriting implemented and evaluated — found to hurt performance, not used in final pipeline |
| **Deployment to cloud** | Deployed on Railway [Tumor Microbiome Assistant](https://literature-rag-production.up.railway.app/) |


## Project components - tested configurations

| Component | What was used | What was tested |
|-----------|--------------|----------------|
| **Knowledge base** | 12 open-access tumor microbiome papers (PDF), chunked into 1000-character chunks with 200 overlap, stored in PostgreSQL with pgvector | `paraphrase-MiniLM-L6-v2`<br>`all-mpnet-base-v2`<br>`all-MiniLM-L6-v2`<br>`allenai-specter`<br>`pritamdeka/S-PubMedBert-MS-MARCO` ✅ |
| **Retrieval pipeline** | Hybrid search (vector + BM25) with cross-encoder reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, `pritamdeka/S-PubMedBert-MS-MARCO` embeddings, GPT-4o-mini for answer generation | Vector search, num_results=5<br>Vector search, num_results=10<br>Hybrid search, num_results=10<br>Hybrid + query rewriting, num_results=10<br>Hybrid + reranking, num_results=10 ✅ |
| **RAG Evaluation** | Sample of 200 questions from ground truth generated with LLM (5 questions per chunk), LLM-as-a-judge for answer quality | `gpt-4o-mini`, default prompt ✅<br>`gpt-4o-mini`, concise prompt<br>`gpt-5.6-terra`, default prompt<br>`gpt-5.6-terra`, concise prompt |
| **User interface** | Flask web app with chat interface, conversation history, 👍/👎 feedback buttons | — |
| **Monitoring** | PostgreSQL monitoring database, Grafana dashboard tracking questions, response time, token usage and user feedback | — |

## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — for running the ingestion notebook
- OpenAI API key

## Local Setup

### 1. Clone the repository
```bash
git clone git clone https://github.com/Mrt905/literature-rag.git
cd literature-rag
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

- Name your PDF files as: `keyword1_keyword2-author-journal-year.pdf`
- Place them in the `data/raw/papers_pdf/` folder
- Run the ingestion notebook:

```bash
uv run jupyter nbconvert --to notebook --execute 01_ingest.ipynb
```

This will:
- Extract text from PDFs
- Chunk documents into 1000-character pieces
- Generate embeddings using `pritamdeka/S-PubMedBert-MS-MARCO`
- Store chunks and embeddings in PostgreSQL

The text search index for hybrid search is created automatically when Docker starts via `init.sql`.

> **Note:** Only run this once

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

## Deployment

The app is deployed on Railway at: [Tumor Microbiome Assistant](https://literature-rag-production.up.railway.app/)

To deploy your own instance:
1. Fork this repository
2. Create a [Railway](https://railway.app) account
3. Create a new project from your GitHub repo — Railway will automatically detect the `Dockerfile`
4. Add a **PostgreSQL** service to your project
5. In the PostgreSQL service → Data tab, run:
```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE TABLE IF NOT EXISTS conversations (
       id SERIAL PRIMARY KEY,
       question TEXT,
       answer TEXT,
       response_time FLOAT,
       input_tokens INTEGER,
       output_tokens INTEGER,
       total_tokens INTEGER,
       feedback INTEGER,
       timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
```
6. Add these environment variables to your app service:
   - `OPENAI_API_KEY` — your OpenAI API key
   - `DATABASE_URL` — reference your Postgres service: `${{Postgres.DATABASE_URL}}`
   - `MONITORING_URL` — same: `${{Postgres.DATABASE_URL}}`
7. Enable **Public Networking** on the PostgreSQL service to get a public connection URL
8. Run `01_ingest.ipynb` using the public Postgres URL to populate the database with your papers
9. Generate a public domain for your app service in Settings → Networking

## Evaluation

> Results are specific to the 12 papers used in this project on tumor microbiome. Results will vary depending on the number, topic, and quality of papers provided.

**Best retrieval model:** 
hybrid search (`pritamdeka/S-PubMedBert-MS-MARCO`for embedding for vector search, best embedding model as determined in evaluation/04_evaluation-ret-exp.ipynb) + reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) - the best performing retrieval model as determined in 05_evaluation-ret-exp2.ipynb
- Hit Rate@10: 0.692
- MRR@10: 0.556

**Best RAG:**
gpt-4o-mini llm model, default prompt (defined in ragbase.py) - best performing RAG system as determined in 06_evaluation-rag.ipynb
- Relevant: 94.0%
- Partly relevant: 4.5%
- Non-relevant: 1.5%
- Average cost per question: 0.0005$

The relatively low retrieval scores are due to all papers being on the same topic (tumor microbiome), making chunks semantically similar and harder to distinguish. 