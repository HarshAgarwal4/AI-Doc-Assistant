from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split into chunks of about 500 to 1,000 characters with overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=['\n\n' , '\n' , ' ' , '']
)


def chunk_page_data(pages: list[dict], filename: str) -> list[dict]:
    chunks = []
    for page in pages:
        page_num = page.get("page_number", 1)
        text = page.get("text", "").strip()

        if not text:
            continue

        for chunk_text in text_splitter.split_text(text):
            chunks.append({
                "text": chunk_text.strip(),
                "filename": filename,
                "page_number": page_num,
            })

    return chunks
