import pymupdf


def read_pdf_pages(file_path: str) -> list[dict]:
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
