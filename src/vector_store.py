import chromadb

class VectorStore:
    def __init__(self, pDirectory: str = "data/chroma"):
        self.client = chromadb.PersistentClient(
            path=pDirectory
        )
        self.collection = self.client.get_or_create_collection(
            name="research_papers"
        )
        
    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str]
    ):
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

