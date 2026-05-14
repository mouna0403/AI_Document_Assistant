"""
test_downloader.py
Tests for multi-format text extraction, including OCR for scanned PDFs
"""

import io

import pandas as pd
import pytest
from PIL import Image, ImageDraw, ImageFont

from AI_Document_Assitant.utils.downloader import extract_text_from_file

# -----------------------------
# Helper functions to create test files
# -----------------------------


def create_fake_pdf(text: str) -> io.BytesIO:
    """Create a fake PDF with text for testing purposes."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, text)
    c.save()
    buffer.seek(0)
    return buffer


def create_fake_scanned_pdf(text: str) -> io.BytesIO:
    """
    Create a PDF containing text as an image to simulate a scanned PDF.
    OCR should be needed to extract the text.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    # Create image with text
    img = Image.new("RGB", (400, 100), color="white")
    d = ImageDraw.Draw(img)
    try:
        # Use a default font
        d.text((10, 10), text, fill="black")
    except Exception:
        d.text((10, 10), text)  # fallback if default font not found

    # Save image to buffer
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    img_reader = ImageReader(img_buffer)

    # Create PDF and embed the image
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawImage(img_reader, 50, 650, width=400, height=100)
    c.save()
    buffer.seek(0)
    return buffer


def create_fake_docx(text: str) -> io.BytesIO:
    from docx import Document

    buffer = io.BytesIO()
    doc = Document()
    doc.add_paragraph(text)
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def create_fake_txt(text: str) -> io.BytesIO:
    buffer = io.BytesIO()
    buffer.write(text.encode("utf-8"))
    buffer.seek(0)
    return buffer


def create_fake_csv(data: list[list[str]]) -> io.BytesIO:
    buffer = io.BytesIO()
    df = pd.DataFrame(data)
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    return buffer


# -----------------------------
# Test class
# -----------------------------


class TestExtractTextFromFile:
    """Test suite for extract_text_from_file function"""

    # --- PDF tests ---
    def test_extract_pdf_text(self):
        fake_pdf = create_fake_pdf("Hello PDF")
        fake_pdf.seek(0)  # Reset position before passing to function
        result = extract_text_from_file(fake_pdf, ocr_lang="eng")
        assert "Hello PDF" in result
        assert isinstance(result, str)

    def test_extract_scanned_pdf_ocr(self):
        """Test that OCR works for scanned PDFs (image-based)"""
        text_content = "Hello OCR PDF"
        fake_pdf = create_fake_scanned_pdf(text_content)
        fake_pdf.seek(0)  # Reset position before passing to function
        result = extract_text_from_file(fake_pdf, ocr_lang="eng")
        # OCR may introduce small artifacts, so check partial match
        assert "Hello" in result or "OCR" in result or "PDF" in result
        assert isinstance(result, str)

    # --- DOCX tests ---
    def test_extract_docx_text(self):
        fake_docx = create_fake_docx("Hello DOCX")
        fake_docx.seek(0)  # Reset position before passing to function
        result = extract_text_from_file(fake_docx)
        assert "Hello DOCX" in result
        assert isinstance(result, str)

    # --- TXT tests ---
    def test_extract_txt_text(self):
        fake_txt = create_fake_txt("Hello TXT")
        fake_txt.seek(0)  # Reset position before passing to function
        result = extract_text_from_file(fake_txt)
        assert "Hello TXT" in result
        assert isinstance(result, str)

    # --- CSV tests ---
    def test_extract_csv_text(self):
        fake_csv = create_fake_csv([["Name", "Age"], ["Alice", "30"], ["Bob", "25"]])
        fake_csv.seek(0)  # Reset position before passing to function
        result = extract_text_from_file(fake_csv)
        assert "Alice" in result
        assert "Bob" in result
        assert isinstance(result, str)

    # --- Edge cases ---
    def test_extract_empty_pdf(self):
        fake_pdf = create_fake_pdf("")
        fake_pdf.seek(0)  # Reset position before passing to function
        result = extract_text_from_file(fake_pdf, ocr_lang="eng")
        # Empty PDF might have some OCR artifacts, just check it doesn't error
        assert isinstance(result, str)

    def test_extract_strips_whitespace_txt(self):
        fake_txt = create_fake_txt("   Text with spaces   ")
        fake_txt.seek(0)  # Reset position before passing to function
        result = extract_text_from_file(fake_txt)
        assert result == result.strip()
