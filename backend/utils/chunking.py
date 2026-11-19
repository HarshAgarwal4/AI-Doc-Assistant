
import logging
import time
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- LOGGING CONFIGURATION ----------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def log_time(step: str, start: float):
    elapsed = time.perf_counter() - start
    logger.info("%s completed in %.4f seconds", step, elapsed)
    return elapsed


# ---------------- CHUNKER CONFIGURATION ----------------

recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

docling_chunker = HybridChunker(
    max_tokens=256,
    merge_peers=True,
    repeat_table_header=True
)


# ---------------- NORMAL CHUNKING ----------------

def chunk_page_data(
    pages: list[dict],
    filename: str
) -> list[dict]:

    total_start = time.perf_counter()
    chunks = []

    logger.info("=" * 60)
    logger.info("STARTING RECURSIVE CHUNKING")
    logger.info("Filename: %s", filename)
    logger.info("Total pages: %d", len(pages))
    logger.info("=" * 60)

    for index, page in enumerate(pages, start=1):

        page_start = time.perf_counter()

        page_num = page.get("page_number", 1)
        text = page.get("text", "").strip()

        logger.debug(
            "Processing page %d | Page number: %s | Characters: %d",
            index, page_num, len(text)
        )

        if not text:
            logger.warning("Page %s is empty. Skipping.", page_num)
            continue

        split_start = time.perf_counter()

        page_chunks = recursive_text_splitter.split_text(text)

        log_time(
            f"Page {page_num} splitting",
            split_start
        )

        for chunk_text in page_chunks:
            chunks.append({
                "text": chunk_text.strip(),
                "filename": filename,
                "page_number": page_num,
            })

        logger.info(
            "Page %s | Chunks: %d | Time: %.4fs",
            page_num,
            len(page_chunks),
            time.perf_counter() - page_start
        )

    total_time = log_time("Total recursive chunking", total_start)

    logger.info("=" * 60)
    logger.info("RECURSIVE CHUNKING SUMMARY")
    logger.info("Total chunks: %d", len(chunks))
    logger.info("Total time: %.4f seconds", total_time)

    if chunks:
        logger.info(
            "Average time per chunk: %.6f seconds",
            total_time / len(chunks)
        )

    logger.info("=" * 60)

    print_chunks(chunks)

    return chunks


# ---------------- DOCLING CHUNKING ----------------

def chunk_with_docling(file_path: str) -> list[dict]:

    total_start = time.perf_counter()
    filename = Path(file_path).name

    logger.info("=" * 60)
    logger.info("STARTING DOCLING CHUNKING")
    logger.info("Filename: %s", file_path)
    logger.info("=" * 60)

    # Step 1: Initialize converter
    step_start = time.perf_counter()

    converter = DocumentConverter()

    log_time("Docling converter initialization", step_start)

    # Step 2: Convert Document (PDF or DOCX)
    step_start = time.perf_counter()

    logger.info("Converting document using Docling...")

    result = converter.convert(file_path)

    doc = result.document

    conversion_time = log_time("Docling document conversion", step_start)

    logger.info(
        "Document conversion completed | Pages: %d",
        len(doc.pages) if doc.pages else 0
    )

    # Step 3: Generate chunks
    step_start = time.perf_counter()

    logger.info("Generating chunks using HybridChunker...")

    chunks = []

    for index, chunk in enumerate(
        docling_chunker.chunk(dl_doc=doc),
        start=1
    ):

        chunk_start = time.perf_counter()

        # Extract page numbers and check if chunk contains table
        page_numbers = set()
        is_table = False
        table_ref = None

        for item in chunk.meta.doc_items:
            for prov in item.prov:
                if prov.page_no is not None:
                    page_numbers.add(prov.page_no)
            if getattr(item, "label", None) == "table" or str(getattr(item, "label", "")).lower() == "table":
                is_table = True
                if hasattr(item, "self_ref"):
                    table_ref = item.self_ref

        # Store all tables with:
        # 1. Heading
        # 2. Table name of file name
        # 3. Table headers
        # 4. Table data
        if is_table:
            headings = chunk.meta.headings or []
            heading_text = " > ".join(headings) if headings else "Table"

            table_item = None
            table_idx = 1
            if table_ref:
                for idx, t in enumerate(doc.tables, start=1):
                    if getattr(t, "self_ref", None) == table_ref:
                        table_item = t
                        table_idx = idx
                        break
            if not table_item and doc.tables:
                table_item = doc.tables[0]

            caption = table_item.caption_text(doc).strip() if table_item else ""
            table_name = caption if caption else f"Table {table_idx}"
            table_title = f"{table_name} of {filename}"

            header_str = ""
            if table_item:
                try:
                    table_md = table_item.export_to_markdown(doc=doc).strip()
                    lines = [l for l in table_md.splitlines() if l.strip()]
                    if len(lines) >= 2 and "|" in lines[0]:
                        header_str = f"{lines[0]}\n{lines[1]}"
                except Exception:  # noqa: BLE001, S110
                    pass

            raw_chunk = chunk.text.strip()
            if raw_chunk.startswith("| - |"):
                raw_chunk = raw_chunk[5:].strip()

            table_parts = [
                f"Heading: {heading_text}",
                f"{table_title}",
            ]
            if header_str:
                table_parts.extend(["Table Headers:", header_str])
            table_parts.extend(["Table Data:", raw_chunk])

            chunk_text = "\n\n".join(table_parts)
        else:
            chunk_text = docling_chunker.contextualize(chunk)

        if not chunk_text.strip():
            logger.warning("Chunk %d is empty. Skipping.", index)
            continue

        chunk_data = {
            "text": chunk_text.strip(),
            "filename": filename,
            "page_number": min(page_numbers) if page_numbers else 1,
        }

        chunks.append(chunk_data)

        logger.debug(
            "Chunk %d | Pages: %s | Characters: %d | Time: %.4fs",
            index,
            sorted(page_numbers),
            len(chunk_text),
            time.perf_counter() - chunk_start
        )

    # ---------------- 1 EXTRA CHUNK: All Headings of File Name ----------------
    headings_list = []
    for item in doc.texts:
        if getattr(item, "label", None) == "section_header" or item.__class__.__name__ == "SectionHeaderItem":
            level = getattr(item, "level", 1) or 1
            prov_pages = [p.page_no for p in item.prov if p.page_no is not None]
            page_info = f" (Page {prov_pages[0]})" if prov_pages else ""
            indent = "  " * max(0, level - 1)
            headings_list.append(f"{indent}- {item.text.strip()}{page_info}")

    if headings_list:
        common_heading = f"All Headings of {filename}"
        headings_body = "\n".join(headings_list)
        split_chunks = recursive_text_splitter.split_text(headings_body)
        for h_text in split_chunks:
            full_text = f"{common_heading}\n\n{h_text.strip()}"
            chunks.append({
                "text": full_text.strip(),
                "filename": filename,
                "page_number": 1,
            })
            logger.info("Added document headings overview chunk | Length: %d", len(full_text))

    chunking_time = log_time("Docling chunk generation", step_start)

    total_time = log_time("Total Docling processing", total_start)

    logger.info("=" * 60)
    logger.info("DOCLING CHUNKING SUMMARY")
    logger.info("Document conversion: %.4fs", conversion_time)
    logger.info("Chunk generation: %.4fs", chunking_time)
    logger.info("Total chunks: %d", len(chunks))
    logger.info("Total processing time: %.4fs", total_time)

    if chunks:
        logger.info(
            "Average time per chunk: %.6f seconds",
            chunking_time / len(chunks)
        )

    logger.info("=" * 60)

    print_chunks(chunks)

    return chunks


# Alias for backward compatibility
chunk_pdf_with_docling = chunk_with_docling


# ---------------- FINAL CHUNK DISPLAY ----------------

def print_chunks(chunks: list[dict]):

    print("\n" + "=" * 80)
    print("FINAL GENERATED CHUNKS")
    print("=" * 80)

    for index, chunk in enumerate(chunks, start=1):

        print(f"\n{'-' * 80}")
        print(f"CHUNK NUMBER : {index}")
        print(f"FILENAME     : {chunk['filename']}")
        print(f"PAGE NUMBER  : {chunk['page_number']}")
        print(f"CHARACTERS   : {len(chunk['text'])}")
        print(f"{'-' * 80}")

        print(chunk["text"])

    print("\n" + "=" * 80)
    print(f"TOTAL CHUNKS: {len(chunks)}")
    print("=" * 80)
