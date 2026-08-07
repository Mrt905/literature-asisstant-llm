from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from ragbase import RAGBaseHistory
import psycopg

load_dotenv()

app = Flask(__name__)

# setup once when app starts
print("Connecting to database...")
conn = psycopg.connect("postgresql://user:pswd@localhost:5432/papers")
print("Connected!")

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


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data["question"]
    
    response = assistant.rag(question)
    
    return jsonify({
        "answer": response.answer,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens
        }
    })

@app.route("/clear", methods=["POST"])
def clear():
    assistant.clear_history()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True)