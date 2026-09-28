import os
from openai import OpenAI


class AnswerGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[dict]
    ) -> str:

        context_parts = []

        for index, item in enumerate(retrieved_chunks, start=1):
            metadata = item["metadata"]

            citation = (
                f"{metadata.get('source', 'Unknown source')}, "
                f"page {metadata.get('page_number', 'unknown')}"
            )

            context_parts.append(
                f"[Passage {index} | {citation}]\n"
                f"{item['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are an academic research assistant.

Answer the user's question using only the supplied passages.

Rules:
1. Do not invent information.
2. If the passages do not contain enough information, say so.
3. Explain technical concepts clearly.
4. Cite supporting passages using their source and page number.
5. Distinguish between what the paper states and your explanation.

User question:
{question}

Retrieved passages:
{context}
"""

        response = self.client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        return response.output_text