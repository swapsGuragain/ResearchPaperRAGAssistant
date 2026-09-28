from src.pdf_processor import extract_pages_from_pdf
from src.chunker import chunk_text
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


def ingest_pdf(pdf_path: str):
    pages = extract_pages_from_pdf(pdf_path)

    documents = []
    metadatas = []
    ids = []

    for page in pages:
        chunks = chunk_text(page["text"])

        for chunk_index, chunk in enumerate(chunks):
            documents.append(chunk)

            metadatas.append(
                {
                    "source": page["source"],
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                }
            )

            ids.append(
                f"{page['source']}_"
                f"{page['page_number']}_"
                f"{chunk_index}"
            )

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.encode_documents(documents)

    vector_store = VectorStore()
    vector_store.add_documents(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )