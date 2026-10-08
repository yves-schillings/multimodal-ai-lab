"""The fallback must handle image pages without replacing usable native text."""
from PIL import Image
from pypdf import PdfWriter
import pytest
from lab import documents


def image_pdf(tmp_path):
    path = tmp_path / 'scan.pdf'
    with Image.new('RGB', (800, 500), 'white') as image:
        image.save(path, 'PDF')
    return path


def test_image_pdf_renders_for_ocr(tmp_path, monkeypatch):
    import pytesseract
    sizes = []
    def ocr(image, timeout):
        sizes.append(image.size)
        assert 0 < timeout <= 45
        return 'Synthetic receipt for a blue bicycle'
    monkeypatch.setattr(pytesseract, 'image_to_string', ocr)
    result = documents.extract_document(image_pdf(tmp_path))
    assert result['text'] == 'Synthetic receipt for a blue bicycle'
    assert result['engine'] == 'pypdf-text+local-tesseract-ocr'
    assert sizes == [(1600, 1000)]


def test_text_page_keeps_native_extraction(tmp_path, monkeypatch):
    from pypdf import PdfReader
    from pypdf._page import PageObject
    path = image_pdf(tmp_path)
    monkeypatch.setattr(PageObject, 'extract_text', lambda self: 'Native receipt text')
    monkeypatch.setattr(documents, '_ocr_pdf_page', lambda *a: pytest.fail('OCR must not replace native text'))
    assert documents.extract_document(path) == {'text': 'Native receipt text', 'engine': 'pypdf-text'}


def test_oversize_pdf_page_rejected_before_render(tmp_path):
    path = tmp_path / 'large.pdf'
    writer = PdfWriter()
    writer.add_blank_page(width=10000, height=10000)
    writer.write(path)
    with pytest.raises(ValueError, match='20 million pixels'):
        documents.extract_document(path)


def test_empty_scan_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, '_ocr_pdf_page', lambda *a: '')
    with pytest.raises(ValueError, match='No text extracted'):
        documents.extract_document(image_pdf(tmp_path))


def test_mixed_pdf_retains_page_order(tmp_path, monkeypatch):
    from pypdf import PdfReader
    from pypdf._page import PageObject
    path = tmp_path / 'mixed.pdf'
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    writer.add_blank_page(width=400, height=300)
    writer.write(path)
    monkeypatch.setattr(PageObject, 'extract_text', lambda p: 'First native page' if p.mediabox.width == 300 else '')
    monkeypatch.setattr(documents, '_ocr_pdf_page', lambda *a: 'Second scanned page')
    assert documents.extract_document(path)['text'] == 'First native page\nSecond scanned page'
