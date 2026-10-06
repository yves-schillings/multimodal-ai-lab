"""Bounded local extraction and immutable originals with reviewed projections."""
from pathlib import Path

MAX_TEXT = 300000


def extract_document(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {'.txt', '.md', '.csv'}:
        try:
            text = path.read_text(encoding='utf-8-sig')
        except UnicodeError:
            raise ValueError('Text documents must use UTF-8 encoding.') from None
        engine = 'utf8'
    elif suffix == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(path)
        if reader.is_encrypted or len(reader.pages) > 100:
            raise ValueError('Use an unencrypted PDF with at most 100 pages.')
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or '')
            if sum(map(len, parts)) > MAX_TEXT:
                raise ValueError('Extracted document exceeds 300000 characters.')
        text, engine = '\n'.join(parts), 'pypdf-text'
    elif suffix in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
        import pytesseract
        from PIL import Image
        with Image.open(path) as image:
            if image.width * image.height > 20000000:
                raise ValueError('Image exceeds 20 million pixels.')
            text = pytesseract.image_to_string(image, timeout=45)
        engine = 'local-tesseract-ocr'
    else:
        raise ValueError('Supported documents: UTF-8 TXT/MD/CSV, PDF, PNG/JPEG/TIFF.')
    if not text.strip():
        raise ValueError('No text extracted. Scanned PDFs require external local OCR preparation.')
    if len(text) > MAX_TEXT:
        raise ValueError('Extracted document exceeds 300000 characters.')
    return {'text': text.strip(), 'engine': engine}
