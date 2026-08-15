from dataclasses import dataclass

@dataclass
class RAGResponse:
    answer: str
    usage: object

INSTRUCTIONS = """
Your task is to answer questions from the provided scientific papers.

Use the context to find relevant information and provide accurate
answers. Using two different sources for the same answer is good, you can try to reason
but make sure that you provide accurate info and source of the information. 
If the answer is not found in the context, respond with "I don't know." 
If the exact answer is not found in the context, but a useful information is, respond with 
"I don't know exactly, but I have some related information that might help" and provide the information.
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:

    def __init__(
        self,
        conn,
        model,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        llm_model="gpt-4o-mini"
    ):
        self.conn = conn
        self.model = model
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.llm_model = llm_model

    def vector_search(self, query, num_results=10):
        """Search by vector similarity"""
        query_vector = self.model.encode(query)
        query_str = "[" + ",".join(str(x) for x in query_vector) + "]"

        rows = self.conn.execute(
            """
            SELECT id, text, filename, author, journal, year,
                1 - (embedding <=> %s::vector) AS score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_str, query_str, num_results)
        ).fetchall()

        return [
            {
                "id": r[0],
                "text": r[1],
                "filename": r[2],
                "author": r[3],
                "journal": r[4],
                "year": r[5],
                "score": r[6]
            }
            for r in rows
        ]

    def text_search(self, query, num_results=10):
        """Search by full text (BM25)"""
        rows = self.conn.execute(
            """
            SELECT id, text, filename, author, journal, year,
                ts_rank(to_tsvector('english', text), plainto_tsquery('english', %s)) AS score
            FROM chunks
            WHERE to_tsvector('english', text) @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, query, num_results)
        ).fetchall()

        return [
            {
                "id": r[0],
                "text": r[1],
                "filename": r[2],
                "author": r[3],
                "journal": r[4],
                "year": r[5],
                "score": r[6]
            }
            for r in rows
        ]

    def rrf(self, result_lists, k=60, num_results=10):
        """Reciprocal Rank Fusion to combine result lists"""
        scores = {}
        docs = {}

        for results in result_lists:
            for rank, doc in enumerate(results):
                key = doc["id"]
                scores[key] = scores.get(key, 0) + 1 / (k + rank)
                docs[key] = doc

        ranked = sorted(scores, key=scores.get, reverse=True)
        return [docs[key] for key in ranked[:num_results]]

    def hybrid_search(self, query, num_results=10):
        """Combine vector and text search with RRF"""
        vector_results = self.vector_search(query, num_results=num_results)
        text_results = self.text_search(query, num_results=num_results)
        return self.rrf([vector_results, text_results], num_results=num_results)

    def search(self, query, num_results=10):
        return self.hybrid_search(query, num_results=num_results)


    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(f"[{doc['author']} {doc['year']} - {doc['filename']}]")
            lines.append(doc["text"])
            lines.append("")

        return "\n".join(lines).strip()  # ← moved outside the loop

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.llm_model,
            input=input_messages
        )

        return response


    def rag(self, query):
        print(f"Original: {query}")
        
        search_results = self.search(query)  
        prompt = self.build_prompt(query, search_results) 
        response = self.llm(prompt)
        return RAGResponse(
            answer=response.output_text,
            usage=response.usage
        )


# RAGBaseEval - Ragbase used for evaluation experiment, table_name added
class RAGBaseEval(RAGBase):

    def __init__(self, table_name="chunks", **kwargs):
        super().__init__(**kwargs)
        self.table_name = table_name

    def search(self, query, num_results=5):
        query_vector = self.model.encode(query)
        query_str = "[" + ",".join(str(x) for x in query_vector) + "]"

        rows = self.conn.execute(
            f"""
            SELECT id, text, filename, author, journal, year,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {self.table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_str, query_str, num_results)
        ).fetchall()

        return [
            {
                "id": r[0],
                "text": r[1],
                "filename": r[2],
                "author": r[3],
                "journal": r[4],
                "year": r[5],
                "similarity": r[6]
            }
            for r in rows
        ]

#RAGBaseHistory - ragbase that stores history of the conversation and uses hybrid search + reranking
from sentence_transformers import CrossEncoder

class RAGBaseHistory(RAGBase):

    def __init__(self, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2", **kwargs):
        super().__init__(**kwargs)
        self.history = []
        self.reranker = CrossEncoder(reranker_model)

    def search(self, query, num_results=10):
        # hybrid search + reranking
        results = self.hybrid_search(query, num_results=20)
        pairs = [(query, doc["text"]) for doc in results]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in reranked[:num_results]]

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm_with_history(prompt)
        self.history.append({
            "question": query,
            "answer": response.output_text
        })
        return RAGResponse(
            answer=response.output_text,
            usage=response.usage
        )

    def llm_with_history(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions}
        ]
        for h in self.history:
            input_messages.append({"role": "user", "content": h["question"]})
            input_messages.append({"role": "assistant", "content": h["answer"]})
        input_messages.append({"role": "user", "content": prompt})
        return self.llm_client.responses.create(
            model=self.llm_model,
            input=input_messages
        )

    def clear_history(self):
        self.history = []