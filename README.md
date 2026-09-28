# ResearchPaperRAGAssistant

A retrieval-augmented generation application that allows users to upload research papers, semantically search relevant passages, and ask technical questions with source- and page-level citations.

# Componenets: Technology 

Programming language: Python

PDF extraction: PyMuPDF

Text processing: Python

Embeddings: Sentence Transformers or an embedding API

Vector database: ChromaDB

LLM: OpenAI API or another compatible LLM provider

Backend: FastAPI

Frontend: Streamlit

Metadata: SQLite or ChromaDB metadata 

Version control: Git/GitHub 

Deployment later: Docker + a cloud platform 

# SYSTEM ARCHITECTURE:

PDF Upload ->
Text Extraction ->
Text Cleaning ->
Chunking ->
Embedding Generation ->
ChromaDB ->
User Question ->
Question Embedding ->
Similarity Search ->
Relevant Passages ->
Prompt Construction ->
LLM Response ->
Answer + Citations

