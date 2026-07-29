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
If the exact asnwer is not found in the context, but a useful information is, respond with 
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

    def search(self, query, num_results=5):

        query_vector = self.model.encode(query)
        query_str = "[" + ",".join(str(x) for x in query_vector) + "]"

        rows = self.conn.execute(
            """
            SELECT id, text, filename, author, journal, year,
                   1 - (embedding <=> %s::vector) AS similarity
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
                "similarity": r[6]
            }
            for r in rows
        ]

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
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return RAGResponse(
            answer = response.output_text,
            usage = response.usage
        )
