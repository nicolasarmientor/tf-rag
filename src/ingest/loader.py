
from pathlib import Path
import re
import nbformat
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUIDE_ROOT = PROJECT_ROOT / "data" / "raw" / "guide"

@dataclass
class Document:
    text: str
    title: str
    source_path: str

def parse_markdown_file(filepath: Path) -> Document:
    raw_text = filepath.read_text(encoding="utf-8")
    
    match = re.search(r"^#\s+(.+)$", raw_text, re.MULTILINE)
    title = match.group(1).strip() if match else filepath.stem

    source_path = str(filepath.relative_to(GUIDE_ROOT))

    return Document(text=raw_text, title=title, source_path=source_path)

def parse_notebook_file(filepath: Path) -> Document:
    notebook = nbformat.read(filepath, as_version=4)

    pieces = []

    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            pieces.append(cell.source)
        elif cell.cell_type == "code":
            code_lines = cell.source.split("\n")
            clean_lines = [line for line in code_lines if not line.strip().startswith(("!", "%"))]
            clean_code = "\n".join(clean_lines)
            if clean_code.strip():
                wrapped = f"```python\n{clean_code}\n```"
                pieces.append(wrapped)
    
    complete_text = "\n\n".join(pieces)

    match = re.search(r"^#\s+(.+)$", complete_text, re.MULTILINE)
    title = match.group(1).strip() if match else filepath.stem

    source_path = str(filepath.relative_to(GUIDE_ROOT))

    return Document(text=complete_text, title=title, source_path=source_path)

def load_documents(guide_root: Path = GUIDE_ROOT) -> list[Document]:
    documents = []

    markdown_files = guide_root.rglob("*.md")
    for filepath in markdown_files:
        document = parse_markdown_file(filepath)
        documents.append(document)

    notebook_files = guide_root.rglob("*.ipynb")
    for filepath in notebook_files:
        document = parse_notebook_file(filepath)
        documents.append(document)
    
    return documents

if __name__ == "__main__":
    docs = load_documents()

    print(f"\nLoaded {len(docs)} documents")
    print("---")
    print(f"Sample title: {docs[0].title}")
    print(f"Sample source_path: {docs[0].source_path}")
    print(f"Sample text (first 200 chars): {docs[0].text[:200]}\n")