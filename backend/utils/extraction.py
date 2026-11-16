from pathlib import Path

import pymupdf


def read_pdf_pages(file_path: str) -> list[dict]:
    """Read a PDF file page by page using PyMuPDF."""
    pages = []
    doc = pymupdf.open(file_path)
    try:
        for index, page in enumerate(doc):
            pages.append({
                "page_number": index + 1,
                "text": page.get_text() or "",
            })
    finally:
        doc.close()

    return pages


def read_docx_with_docling(file_path: str) -> list[dict]:
    """Read a DOCX file using Docling, preserving accurate page numbers."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document

    # 1. If Docling tracked page numbers directly
    pages = []
    if doc.pages:
        for page_no in sorted(doc.pages.keys()):
            page_text = doc.export_to_markdown(page_no=page_no)
            if page_text and page_text.strip():
                pages.append({"page_number": page_no, "text": page_text.strip()})
        if pages:
            return pages
    try:
        import docx

        docx_doc = docx.Document(file_path)
        current_paragraphs = []
        page_num = 1

        for p in docx_doc.paragraphs:
            xml = p._p.xml
            if ("w:br" in xml or "lastRenderedPageBreak" in xml) and current_paragraphs:
                pages.append({
                    "page_number": page_num,
                    "text": "\n\n".join(current_paragraphs),
                })
                page_num += 1
                current_paragraphs = []

            text = p.text.strip()
            if text:
                current_paragraphs.append(text)

        if current_paragraphs:
            pages.append({
                "page_number": page_num,
                "text": "\n\n".join(current_paragraphs),
            })

        if len(pages) > 1:
            return pages
    except Exception:  # noqa: BLE001, S110
        pass

    # 3. Fallback: Full document as page 1
    markdown = doc.export_to_markdown()
    return [{"page_number": 1, "text": markdown or ""}]


def read_document_pages(file_path: str) -> list[dict]:
    """
    Read document pages based on file type:
    - .docx uses Docling (with accurate page numbers)
    - .pdf uses PyMuPDF (fast, page by page)
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".docx":
        return read_docx_with_docling(file_path)
    return read_pdf_pages(file_path)
