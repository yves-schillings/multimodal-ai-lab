"""Bounded local extraction and immutable originals with reviewed projections."""
from pathlib import Path
from threading import Lock
from time import monotonic

MAX_TEXT = 300000
MAX_IMAGE_PIXELS = 20000000
PDF_OCR_SECONDS = 90
# PDFium calls must not overlap, including calls on different documents.
PDF_RENDER_LOCK = Lock()


def _ocr_pdf_page(path, index, timeout):
    import pypdfium2 as pdfium
    import pytesseract
    with PDF_RENDER_LOCK:
        with pdfium.PdfDocument(path) as document:
            page = document[index]
            try:
                width, height = page.get_size()
                scale = 2.0
                if width * height * scale * scale > MAX_IMAGE_PIXELS:
                    raise ValueError('Rendered PDF page exceeds 20 million pixels.')
                bitmap = page.render(scale=scale)
                try:
                    image = bitmap.to_pil().copy()
                finally:
                    bitmap.close()
            finally:
                page.close()
    try:
        return pytesseract.image_to_string(image, timeout=timeout)
    finally:
        image.close()


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
        started = monotonic()
        ocr_pages = 0
        for index, page in enumerate(reader.pages):
            extracted = page.extract_text() or ''
            if not extracted.strip():
                remaining = PDF_OCR_SECONDS - (monotonic() - started)
                if remaining <= 0:
                    raise ValueError('PDF extraction exceeded the 90-second OCR budget.')
                extracted = _ocr_pdf_page(path, index, min(45, remaining))
                ocr_pages += 1
            parts.append(extracted)
            if sum(map(len, parts)) > MAX_TEXT:
                raise ValueError('Extracted document exceeds 300000 characters.')
        text, engine = '\n'.join(parts), 'pypdf-text'
        if ocr_pages:
            engine = 'pypdf-text+local-tesseract-ocr'
    elif suffix in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
        import pytesseract
        from PIL import Image
        with Image.open(path) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError('Image exceeds 20 million pixels.')
            text = pytesseract.image_to_string(image, timeout=45)
        engine = 'local-tesseract-ocr'
    else:
        raise ValueError('Supported documents: UTF-8 TXT/MD/CSV, PDF, PNG/JPEG/TIFF.')
    if not text.strip():
        raise ValueError('No text extracted. Check the source quality and local OCR installation.')
    if len(text) > MAX_TEXT:
        raise ValueError('Extracted document exceeds 300000 characters.')
    return {'text': text.strip(), 'engine': engine}
