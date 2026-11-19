
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

    print(f"\n[cyan]{'=' * 60}[/cyan]")
    print(f"[cyan][read_docx_with_docling][/cyan] Processing file: {file_path}")

    # Step 1: Docling conversion
    print("[cyan][read_docx_with_docling][/cyan] [Step 1] Converting DOCX with Docling...")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document

    docling_page_count = len(doc.pages) if getattr(doc, "pages", None) else 0
    print(f"[cyan][read_docx_with_docling][/cyan] Docling detected {docling_page_count} page(s) in document.")

    pages = []
    if getattr(doc, "pages", None):
        for page_no in sorted(doc.pages.keys()):
            page_items = [
                item.text for item in doc.texts
                if any(p.page_no == page_no for p in getattr(item, "prov", []) if p.page_no is not None)
            ]
            for t in doc.tables:
                if any(p.page_no == page_no for p in getattr(t, "prov", []) if p.page_no is not None):
                    try:
                        t_md = t.export_to_markdown(doc=doc)
                        if t_md:
                            page_items.append(t_md)
                    except Exception:  # noqa: BLE001, S110
                        pass

            if page_items:
                pages.append({"page_number": page_no, "text": "\n\n".join(page_items).strip()})

        if pages:
            print(f"[green][read_docx_with_docling][/green] Successfully extracted {len(pages)} page(s) via Docling provenance.")
            print(f"[cyan]{'=' * 60}[/cyan]\n")
            return pages
        print("[yellow][read_docx_with_docling][/yellow] Docling pages container existed but yielded no text items.")

    # Step 2: Fallback to python-docx page break detection
    print("[yellow][read_docx_with_docling][/yellow] [Step 2] Falling back to python-docx page break detection...")
    try:
        import docx

        docx_doc = docx.Document(file_path)
        print(f"[yellow][read_docx_with_docling][/yellow] python-docx loaded document with {len(docx_doc.paragraphs)} paragraph(s).")
        current_paragraphs = []
        page_num = 1
        page_breaks_found = 0

        for idx, p in enumerate(docx_doc.paragraphs, start=1):
            xml = p._p.xml
            if ("w:br" in xml or "lastRenderedPageBreak" in xml) and current_paragraphs:
                page_breaks_found += 1
                pages.append({
                    "page_number": page_num,
                    "text": "\n\n".join(current_paragraphs),
                })
                print(f"[yellow][read_docx_with_docling][/yellow] Page break #{page_breaks_found} found at paragraph {idx}. Page {page_num} finalized with {len(current_paragraphs)} paragraph(s).")
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
            print(f"[yellow][read_docx_with_docling][/yellow] Final page {page_num} stored with {len(current_paragraphs)} paragraph(s).")

        if len(pages) > 1:
            print(f"[green][read_docx_with_docling][/green] Successfully extracted {len(pages)} page(s) via python-docx page breaks.")
            print(f"[cyan]{'=' * 60}[/cyan]\n")
            return pages

        print(f"[yellow][read_docx_with_docling][/yellow] Only {len(pages)} page found via python-docx (no multi-page breaks detected).")

    except Exception as err:  # noqa: BLE001
        print(f"[red][read_docx_with_docling][/red] python-docx extraction encountered error: {err}")

    # Step 3: Fallback - Full document as page 1
    print("[yellow][read_docx_with_docling][/yellow] [Step 3] Exporting full document text as Page 1 fallback...")
    markdown = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else ""
    final_text = (markdown or "").strip()
    print(f"[green][read_docx_with_docling][/green] Exported single-page markdown ({len(final_text)} characters).")
    print(f"[cyan]{'=' * 60}[/cyan]\n")
    return [{"page_number": 1, "text": final_text}]


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
