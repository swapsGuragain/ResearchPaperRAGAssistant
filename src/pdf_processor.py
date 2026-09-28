from pathlib import Path
import pymupdf

def extract_pages_from_pdf(pdf_path: str)->list[dict]:
    # Extract text from a PDF while preserving page-level metadata.

    pdf_path = Path(pdf_path)
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start = 1):
        text = page.get_text('text').strip()

        if text:
            pages.append(
                {
                    "text": text,
                    "page_number": page_number,
                    "source": pdf_path.name,
                }
            )

    document.close()
    return pages