import os
from pypdf import PdfReader


def extract_text_from_file(file_path: str) -> str:
    """
    Extracts readable text from an uploaded file (PDF or plain text).
    Handles real-world PDFs including multi-column layouts.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _extract_pdf_text(file_path)
    else:
        # Plain text / DOCX fallback — read as UTF-8
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def _extract_pdf_text(file_path: str) -> str:
    """
    Uses pypdf to extract text from a PDF.
    Falls back to raw character extraction if the standard method yields too little text.
    """
    try:
        reader = PdfReader(file_path)

        if len(reader.pages) == 0:
            raise ValueError("PDF contains no readable pages.")

        extracted_pages = []
        for page_idx, page in enumerate(reader.pages):
            # Standard text extraction
            text = page.extract_text(extraction_mode="layout") or ""

            # Fallback: try plain extraction if layout mode returns nothing
            if not text.strip():
                text = page.extract_text() or ""

            if text.strip():
                extracted_pages.append(f"--- Page {page_idx + 1} ---\n{text.strip()}")

        full_text = "\n\n".join(extracted_pages)

        if not full_text.strip():
            raise ValueError(
                "Could not extract any text from this PDF. "
                "The file may be a scanned image-only PDF (OCR not supported yet)."
            )

        return full_text

    except Exception as e:
        error_msg = str(e)
        if "Could not extract" in error_msg or "no readable" in error_msg:
            raise
        raise ValueError(f"PDF parsing failed: {error_msg}")
