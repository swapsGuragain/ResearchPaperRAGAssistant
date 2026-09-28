def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150 ) -> list[str]:
    # Split text into overlapping word-based chunks.
    
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks 