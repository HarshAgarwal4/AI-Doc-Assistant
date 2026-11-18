
from pathlib import Path

import pymupdf
from fastapi import HTTPException
from rich import print


def read_pdf_pages(file_path: str) -> dict:
    """Read PDF using the appropriate parsing strategy."""

    with pymupdf.open(file_path) as pdf:
        total_pages = len(pdf)

    print(f"Total PDF pages: {total_pages}")

    # PDFs <= 15 pages -> Docling
    if total_pages <= 15:
        print("[green]Using Docling[/green]")
        return read_with_docling(file_path)

    # PDFs > 15 pages -> PyMuPDF
    print("[yellow]Using PyMuPDF[/yellow]")

    pages = []

    with pymupdf.open(file_path) as pdf:
        for index, page in enumerate(pdf):
            text = page.get_text("text", sort=True).strip()

            if text:
                pages.append({
                    "page_number": index + 1,
                    "text": text,
                })

    # Normal text-based PDF
    if pages:
        return pages

    # Completely scanned PDF
    print("[yellow]Scanned PDF detected[/yellow]")

    if total_pages > 25:
        raise HTTPException(
            status_code=400,
            detail="Document unprocessable"
        )

    # Scanned PDF <= 25 pages -> Docling
    return read_with_docling(file_path)


def read_with_docling(file_path: str) -> list[dict]:
    """Parse document using Docling."""

    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(file_path)

    doc = result.document
    pages = []

    if doc.pages:
        for page_no in sorted(doc.pages.keys()):
            page_items = [
                item.text for item in doc.texts
                if any(p.page_no == page_no for p in item.prov if p.page_no is not None)
            ]
            for t in doc.tables:
                if any(p.page_no == page_no for p in t.prov if p.page_no is not None):
                    try:
                        t_md = t.export_to_markdown(doc=doc)
                        if t_md:
                            page_items.append(t_md)
                    except Exception:  # noqa: BLE001, S110
                        pass

            if page_items:
                pages.append({
                    "page_number": page_no,
                    "text": "\n\n".join(page_items).strip()
                })

    if not pages:
        markdown = doc.export_to_markdown()

        if markdown and markdown.strip():
            pages.append({
                "page_number": 1,
                "text": markdown.strip()
            })

    if not pages:
        raise HTTPException(
            status_code=400,
            detail="Document unprocessable"
        )

    return pages


def read_docx_with_docling(file_path: str) -> list[dict]:
    """Read a DOCX file using Docling, preserving accurate page numbers."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document

    pages = []
    if doc.pages:
        for page_no in sorted(doc.pages.keys()):
            page_items = [
                item.text for item in doc.texts
                if any(p.page_no == page_no for p in item.prov if p.page_no is not None)
            ]
            for t in doc.tables:
                if any(p.page_no == page_no for p in t.prov if p.page_no is not None):
                    try:
                        t_md = t.export_to_markdown(doc=doc)
                        if t_md:
                            page_items.append(t_md)
                    except Exception:  # noqa: BLE001, S110
                        pass

            if page_items:
                pages.append({"page_number": page_no, "text": "\n\n".join(page_items).strip()})
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

    markdown = doc.export_to_markdown()
    return [{"page_number": 1, "text": (markdown or "").strip()}]


def read_document_pages(file_path: str) -> list[dict]:
    """Route documents based on file extension."""

    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return read_pdf_pages(file_path)

    if ext == ".docx":
        return read_docx_with_docling(file_path)

    raise HTTPException(
        status_code=400,
        detail="Unsupported document format"
    )
