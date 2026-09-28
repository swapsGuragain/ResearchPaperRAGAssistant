from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedding_model.encode_query(question)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):
            retrieved.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return retrieved