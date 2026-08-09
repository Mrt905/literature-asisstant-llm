-- create monitoring database
CREATE DATABASE monitoring;

-- connect to monitoring database
\c monitoring;

-- create conversations table
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    question TEXT,
    answer TEXT,
    response_time FLOAT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    feedback INTEGER,  -- 1 = thumbs up, -1 = thumbs down, NULL = no feedback
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);