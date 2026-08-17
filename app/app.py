import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from ragbase import RAGBaseHistory
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import psycopg
import time

load_dotenv()

app = Flask(__name__)

# connect to papers database
print("Connecting to papers database...")
papers_url = os.getenv("DATABASE_URL", "postgresql://user:pswd@localhost:5432/papers")
conn = psycopg.connect(papers_url)
print("Connected to papers database!")

# connect to monitoring database
print("Connecting to monitoring database...")
monitoring_url = os.getenv("MONITORING_URL", "postgresql://user:pswd@localhost:5432/monitoring")
monitoring_conn = psycopg.connect(monitoring_url)
print("Connected to monitoring database!")

print("Loading embedding model...")
embedding_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
print("Model loaded!")

openai_client = OpenAI()

assistant = RAGBaseHistory(
    conn=conn,
    model=embedding_model,
    llm_client=openai_client,
    llm_model="gpt-4o-mini"
)

def log_conversation(question, answer, response_time, usage):
    """Log conversation to monitoring database"""
    try:
        cursor = monitoring_conn.execute(
            """
            INSERT INTO conversations 
            (question, answer, response_time, input_tokens, output_tokens, total_tokens)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (question, answer, response_time,
             usage.input_tokens, usage.output_tokens, usage.total_tokens)
        )
        monitoring_conn.commit()
        return cursor.fetchone()[0]  # ← return the ID
    except Exception as e:
        print(f"Error logging conversation: {e}")
        monitoring_conn.rollback()
        return None
    
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data["question"]
    
    start_time = time.time()
    response = assistant.rag(question)
    response_time = time.time() - start_time
    
    conversation_id = log_conversation(
        question=question,
        answer=response.answer,
        response_time=response_time,
        usage=response.usage
    )
    
    return jsonify({
        "id": conversation_id,  # ← add this
        "answer": response.answer,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens
        },
        "response_time": round(response_time, 2)
    })


@app.route("/clear", methods=["POST"])
def clear():
    assistant.clear_history()
    return jsonify({"status": "cleared"})

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    conversation_id = data["id"]
    feedback_value = data["feedback"]  # 1 or -1
    
    try:
        monitoring_conn.execute(
            "UPDATE conversations SET feedback = %s WHERE id = %s",
            (feedback_value, conversation_id)
        )
        monitoring_conn.commit()
        return jsonify({"status": "ok"})
    except Exception as e:
        monitoring_conn.rollback()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)