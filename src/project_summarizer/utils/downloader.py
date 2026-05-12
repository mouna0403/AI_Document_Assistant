"""
downloader.py
Utility for extracting text content from uploaded PDFs.
"""

"""
downloader.py
Utility for extracting text content from uploaded files (PDF, DOCX, TXT, CSV).
"""

import io

import pandas as pd
import pytesseract
from docx import Document
from pdf2image import convert_from_bytes
from PIL import Image
from pypdf import PdfReader


def extract_text_from_file(file, ocr_lang="fra") -> str:
    """
    Extract text from various file types.
    Supported formats:
        - PDF (including scanned PDFs using OCR)
        - DOCX (Microsoft Word)
        - TXT (plain text)
        - CSV (comma-separated values)
    Args:
        file: a file-like object (BytesIO or UploadedFile)
        ocr_lang: language for OCR (default: 'fra' for French)
    Returns:
        str: extracted text
    """

    # FIX MINIMAL
    content = file.read() if hasattr(file, "read") else file

    def detect_file_type(data: bytes) -> str:
        if data[:4] == b"%PDF":
            return "pdf"
        elif data[:2] == b"PK":
            return "docx"
        elif b"," in data[:200] and b"\n" in data[:200]:
            return "csv"
        else:
            return "txt"

    file_type = detect_file_type(content)

    # --- PDF files ---
    if file_type == "pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            text = ""

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""

                if len(page_text.strip()) > 50:
                    text += f"\n--- Page {i+1} ---\n{page_text}"
                else:
                    try:
                        images = convert_from_bytes(
                            content, first_page=i + 1, last_page=i + 1, dpi=300
                        )
                        if images:
                            ocr_text = pytesseract.image_to_string(
                                images[0], lang=ocr_lang
                            )
                            text += f"\n--- Page {i+1} (OCR) ---\n{ocr_text}"
                    except Exception as e:
                        text += f"\n--- Page {i+1} (Error) ---\nCould not extract text: {str(e)}\n"

            return text.strip()

        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    # --- DOCX files ---
    elif file_type == "docx":
        try:
            doc = Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs])
            return text.strip()
        except Exception as e:
            return f"Error processing DOCX: {str(e)}"

    # --- CSV files ---
    elif file_type == "csv":
        try:
            df = pd.read_csv(io.BytesIO(content))
            return df.to_string(index=False)
        except Exception as e:
            return f"Error processing CSV: {str(e)}"

    # --- TXT files ---
    else:
        try:
            return content.decode("utf-8", errors="ignore").strip()
        except Exception as e:
            return f"Error processing TXT: {str(e)}"
