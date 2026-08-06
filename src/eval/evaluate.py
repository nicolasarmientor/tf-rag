import os
from anthropic import Anthropic

from src.eval.testset import EVAL_QUESTIONS
from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def check_retrieval(question_data: dict, retrieved_chunks: list[dict]) -> bool:
    expected_source = question_data["expected_source"]

    if expected_source is None:
        return len(retrieved_chunks) == 0 or all(
            c["distance"] > 1.3 for c in retrieved_chunks
        )

    retrieved_sources = {chunk["source_path"] for chunk in retrieved_chunks}

    return expected_source in retrieved_sources

def judge_answer(question: str, generated_answer: str, expected_summary: str) -> bool:
    judge_prompt = f"""Question: {question}

Reference answer (for grading reference only — the generated answer does not need to match this wording): {expected_summary}

Generated answer: {generated_answer}

Does the generated answer correctly and accurately address the question, containing the key facts, even if phrased differently, more detailed, or focused on a slightly different aspect than the reference? Answer "yes" if the generated answer is factually correct and grounded, even if incomplete relative to the reference. Answer "no" only if it contains incorrect information or fails to address the question. Answer with only "yes" or "no"."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": judge_prompt}],
    )

    verdict = response.content[0].text.strip().lower()
    return verdict.startswith("yes")

def run_evaluation():
    retrieval_correct = 0
    answer_correct = 0
    total = len(EVAL_QUESTIONS)

    results = []

    for q in EVAL_QUESTIONS:
        chunks = retrieve(q["question"], top_k=5)
        retrieval_ok = check_retrieval(q, chunks)

        if q["expected_source"] is None:
            gen_result = generate_answer(q["question"], chunks)
            answer_ok = "don't have enough information" in gen_result["answer"].lower()
        else:
            gen_result = generate_answer(q["question"], chunks)
            answer_ok = judge_answer(q["question"], gen_result["answer"], q["expected_answer_summary"])

        retrieval_correct += retrieval_ok
        answer_correct += answer_ok

        results.append({
            "question": q["question"],
            "retrieval_ok": retrieval_ok,
            "answer_ok": answer_ok,
            "generated_answer": gen_result["answer"],
        })

        print(f"[{'✓' if retrieval_ok else '✗'} retrieval] [{'✓' if answer_ok else '✗'} answer] {q['question']}")

    print(f"\nRetrieval accuracy: {retrieval_correct}/{total} ({retrieval_correct/total:.0%})")
    print(f"Answer accuracy: {answer_correct}/{total} ({answer_correct/total:.0%})")

    return results


if __name__ == "__main__":
    results = run_evaluation()
    print("\n--- Failed answers, for manual review ---")
    for r in results:
        if not r["answer_ok"]:
            print(f"\nQ: {r['question']}")
            print(f"Generated: {r['generated_answer']}")