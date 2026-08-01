import fitz  # pymupdf
import os
import psycopg
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# loading pdf papers
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def parse_filename(filename):
    name = filename.replace(".pdf", "")
    
    # split by "-"
    parts = name.split("-")
    
    return {
        "keywords": parts[0].replace("_", " "),
        "author": parts[1],
        "journal": parts[2],
        "year": parts[3]
    }

def load_all_papers(folder):
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            text = extract_text_from_pdf(path)
            metadata = parse_filename(filename)
            documents.append({
                "text": text,
                "filename": filename,
                "keywords": metadata["keywords"],
                "author": metadata["author"],
                "journal": metadata["journal"],
                "year": metadata["year"]
            })
    return documents

## chunking 
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def chunk_documents(documents, chunk_size=1000, overlap=200):
    chunks = []
    for doc in documents:
        doc_chunks = chunk_text(doc["text"], chunk_size, overlap)
        for i, chunk in enumerate(doc_chunks):
            chunks.append({
                "text": chunk,
                "chunk_id": i,
                "filename": doc["filename"],
                "author": doc["author"],
                "journal": doc["journal"],
                "year": doc["year"],
                "keywords": doc["keywords"]
            })
    return chunks

## insert into postgres

def vec_to_str(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"

def setup_database(conn, table_name="chunks", embedding_dim=384):
    """Create the chunks table if it doesn't exist"""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"""
        CREATE TABLE {table_name} (
            id SERIAL PRIMARY KEY,
            text TEXT,
            chunk_id INTEGER,
            filename TEXT,
            author TEXT,
            journal TEXT,
            year TEXT,
            keywords TEXT,
            embedding vector({embedding_dim})
        )
    """)
    conn.commit()



def insert_chunks(conn, chunks, model, table_name="chunks"):
    """Embed and insert chunks into Postgres"""
    for chunk in tqdm(chunks):
        vector = model.encode(chunk["text"])
        vector_str = vec_to_str(vector)
        
        conn.execute(
            f"""
            INSERT INTO {table_name} (text, chunk_id, filename, author, journal, year, keywords, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
            """,
            (chunk["text"], chunk["chunk_id"], chunk["filename"],
             chunk["author"], chunk["journal"], chunk["year"],
             chunk["keywords"], vector_str)
        )
    conn.commit()