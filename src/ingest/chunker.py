
import re
import tiktoken
from src.ingest.loader import Document, load_documents
from dataclasses import dataclass

TARGET_CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50

encoding = tiktoken.get_encoding("cl100k_base")
@dataclass
class Chunk:
    text: str
    chunk_id: str
    source_path: str
    title:str

def find_code_block_spans(text:str) -> list[tuple[int, int]]:
    pattern = r"```.*?\n.*?```"
    spans = []

    for match in re.finditer(pattern, text, re.DOTALL):
        spans.append((match.start(), match.end()))
    
    return spans

def find_chunk_end(text:str, start:int, code_spans: list[tuple[int,int]]) -> int:
    potential_end = start

    while potential_end < len(text):
        potential_end += 1
        temp_chunk = text[start:potential_end]
        token_count = len(encoding.encode(temp_chunk))

        if token_count >= TARGET_CHUNK_TOKENS:
            break

    for span_start, span_end in code_spans:
        if span_start < potential_end < span_end:
            potential_end = span_end

    return min(potential_end, len(text))

def find_overlap_start(text:str, chunk_end: int) -> int:
    potential_start = chunk_end

    while potential_start > 0:
        potential_start -= 1
        temp_overlap = text[potential_start:chunk_end]
        token_count = len(encoding.encode(temp_overlap))

        if token_count >= OVERLAP_TOKENS:
            break

    return potential_start

def chunk_text(text: str, code_spans: list[tuple[int, int]]) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = find_chunk_end(text, start, code_spans)
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break

        start = find_overlap_start(text, end)

    return chunks

def chunk_document(document: Document) -> list[Chunk]:
    spans = find_code_block_spans(document.text)
    raw_chunks = chunk_text(document.text, spans)

    result = []

    for i, chunk_string in enumerate(raw_chunks):
        chunk_id = f"{document.source_path}_{i}"
        chunk = Chunk(
            text=chunk_string,
            chunk_id=chunk_id,
            source_path=document.source_path,
            title=document.title
        )
        result.append(chunk)

    return result


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    all_chunks = []

    for document in documents:
        document_chunks = chunk_document(document)
        all_chunks.extend(document_chunks)
    
    return all_chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"Total chunks: {len(chunks)}")
    print(f"Sample chunk_id: {chunks[0].chunk_id}")
    print(f"Sample source_path: {chunks[0].source_path}")