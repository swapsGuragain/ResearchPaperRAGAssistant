import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.pdf_processor import extract_pages_from_pdf
from src.chunker import chunk_text
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import AnswerGenerator


load_dotenv()

st.set_page_config(
    page_title="Research Paper RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper RAG Assistant")
st.write(
    "Upload research papers and ask questions grounded in their content."
)

uploaded_file = st.file_uploader(
    "Upload a PDF research paper",
    type=["pdf"]
)

if uploaded_file is not None:
    paper_directory = Path("data/papers")
    paper_directory.mkdir(parents=True, exist_ok=True)

    pdf_path = paper_directory / uploaded_file.name

    with open(pdf_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    if st.button("Index Paper"):
        with st.spinner("Processing paper..."):
            pages = extract_pages_from_pdf(str(pdf_path))

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

        st.success("Paper indexed successfully.")

question = st.text_input(
    "Ask a question about the indexed paper:"
)

if st.button("Ask") and question:
    with st.spinner("Searching and generating answer..."):
        retriever = Retriever()
        retrieved_chunks = retriever.retrieve(question, top_k=5)

        generator = AnswerGenerator()
        answer = generator.generate_answer(
            question,
            retrieved_chunks
        )

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved passages")

    for index, item in enumerate(retrieved_chunks, start=1):
        metadata = item["metadata"]

        with st.expander(
            f"Passage {index} — "
            f"{metadata.get('source')} — "
            f"Page {metadata.get('page_number')}"
        ):
            st.write(item["text"])