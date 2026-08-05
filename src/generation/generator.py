
from dotenv import load_dotenv
from anthropic import Anthropic
import markdown as md_lib

load_dotenv()

client = Anthropic()

SYSTEM_PROMPT = """You are a documentation assistant that answers questions about TensorFlow based solely on the provided context.

Rules:
- Only use information from the provided context to answer.
- If the context does not contain enough information to answer, say "I don't have enough information in the provided documentation to answer that".
- When you answer, mention which source(s) you used, by their title.
- Do not use any outside knowledge about TensorFlow beyond what is in the context.
- Do not use markdown formatting (no #, **, bullet points with -, etc). Write plain sentences and pragraphs only.
"""

def build_prompt(question: str, retrieved_chunks:list[dict]) -> str:
    context_sections = []

    for chunk in retrieved_chunks:
        section = f"Source: {chunk['title']} ({chunk['source_path']})\n{chunk['text']}"
        context_sections.append(section)

    context_text = "\n\n---\n\n".join(context_sections)

    prompt = f"""Context:
{context_text}

Question: {question}

Answer based only on the context above."""

    return prompt

def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    if not retrieved_chunks:
        return {
            "answer": "I don't have enough information in the provided documentation to answer that.",
            "sources": [],
            "model": "claude-haiku-4-5-20251001",
            "input_tokens": 0,
            "output_tokens": 0
        }

    prompt = build_prompt(question, retrieved_chunks)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer_text = response.content[0].text
    answer_html = md_lib.markdown(answer_text, extensions=["fenced_code"])

    seen = set()
    sources = []
    for chunk in retrieved_chunks:
        key = chunk["source_path"]
        if key not in seen:
            seen.add(key)
            sources.append({'title': chunk["title"], "source_path": chunk["source_path"]})


    return {
        "answer": answer_html,
        "sources": sources,
        "model": "claude-haiku-4-5",
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }


if __name__ == "__main__":
    from src.retrieval.retriever import retrieve

    question = "What is Keras sequential model?"
    chunks = retrieve(question, top_k=3)
    result = generate_answer(question, chunks)

    print(f"Answer: {result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f" - {s['title']} ({s['source_path']})")