# Tumor Microbiome Literature Assistant

A RAG (Retrieval-Augmented Generation) assistant that answers questions based on scientific literature about tumor microbiome. The system uses vector search to find relevant chunks from uploaded papers and generates accurate, source-grounded answers using an LLM.

## How it works

1. PDF papers are loaded, chunked and embedded using `pritamdeka/S-PubMedBert-MS-MARCO` — a biomedical embedding model
2. Embeddings are stored in PostgreSQL with pgvector extension
3. At query time, the most similar chunks are retrieved and passed to the LLM as context
4. The LLM generates an answer grounded in the provided literature

## Setup

### Prerequisites
- Python 3.13+
- Docker Desktop
- OpenAI API key

### 1. Clone the repository
```bash
git clone https://github.com/your-username/literature-assistant-llm.git
cd literature-assistant-llm
```

### 2. Install dependencies
```bash
uv add pymupdf "psycopg[binary]" sentence-transformers openai python-dotenv flask pandas tqdm numpy
```

### 3. Set up environment variables
Create a `.env` file in the root folder:
```bash
OPENAI_API_KEY=your_api_key_here
```

### 4. Start the database
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

### 5. Add papers
- Name your PDF files as: `keyword1_keyword2-author-journal-year.pdf`
- Place them in the `papers_pdf/` folder

### 6. Ingest papers
Run `01_ingest.ipynb` to load, chunk, embed and store papers in the database.

Only needs to be run once, or when adding new papers.

## Running the app
```bash
uv run python app.py
```

Then open `http://localhost:5000` in your browser.

## Evaluation

The project includes full retrieval and RAG evaluation:

- **Retrieval evaluation** (`03_evaluation-ret.ipynb`, `04_evaluation-ret-exp.ipynb`) — compares embedding models using Hit Rate and MRR
- **RAG evaluation** (`05_evaluation-rag.ipynb`) — evaluates answer quality using LLM-as-a-judge

### Results

Note: These results are specific to the 12 papers used in this project on tumor microbiome. Results will vary depending on the number, topic, and quality of papers provided.

Best embedding model: `pritamdeka/S-PubMedBert-MS-MARCO`
- Hit Rate@10: 0.604
- MRR@10: 0.380

RAG answer quality:
- Relevant: 82.5%
- Partly relevant: 17.0%
- Non-relevant: 0.5%