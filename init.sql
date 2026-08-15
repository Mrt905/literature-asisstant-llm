-- existing monitoring database setup
CREATE DATABASE monitoring;

\c monitoring;

CREATE TABLE conversations (
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

\c papers;

-- add full text search index to chunks table
CREATE INDEX IF NOT EXISTS chunks_fts_idx 
ON chunks USING gin(to_tsvector('english', text));

