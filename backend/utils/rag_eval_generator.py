import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure backend root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Ensure model files can be loaded if current working directory is outside backend
_orig_cwd = os.getcwd()
try:
    if not (Path(_orig_cwd) / "linear_svm_model.pkl").exists() and (BACKEND_DIR / "linear_svm_model.pkl").exists():
        os.chdir(BACKEND_DIR)
    from utils.llm import generate
    from utils.logger import logger
    from utils.prompt_parser import parse_prompt
    from utils.retriever import search
finally:
    os.chdir(_orig_cwd)


async def run_single_eval(item: dict, provider: str = "Groq", k: int = 5) -> dict:
    """
    Directly runs RAG on a single question:
    1. Retrieves top-k chunks from the retriever for context.
    2. Streams and collects the full answer from generate().
    3. Returns a dictionary matching the evaluation schema.
    """
    question = item.get("question", "").strip()
    ground_truth = item.get("ground_truth") or item.get("grounded_truth", "").strip()

    # 1. Fetch retrieval contexts
    contexts = []
    try:
        clean_question = parse_prompt(question)
        raw_chunks = await search(q=clean_question, k=k)
        contexts = [
            chunk.get("text", "").strip()
            for chunk in raw_chunks
            if chunk.get("text")
        ]
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to retrieve contexts for question '%s': %s", question, e)

    # 2. Generate answer directly using generate()
    answer_chunks = []
    try:
        async for event in generate(question=question, provider=provider):
            for line in event.splitlines():
                if line.startswith("data:"):
                    try:
                        payload = json.loads(line[5:].strip())
                        if payload.get("text"):
                            answer_chunks.append(payload["text"])
                    except Exception:  # noqa: BLE001, S110
                        pass
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to generate answer for question '%s': %s", question, e)

    answer = "".join(answer_chunks).strip()

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }


async def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on test questions")
    parser.add_argument(
        "--input",
        type=str,
        default=str(BACKEND_DIR / "rag_eval_questions.json"),
        help="Path to input test questions JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BACKEND_DIR / "rag_testes_questions.json"),
        help="Path to output evaluated questions JSON file",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="Ollama",
        choices=["Ollama", "Groq", "Google", "Mistral"],
        help="LLM provider to use for generation",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of retrieved chunks for context",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of questions to process",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # Fallback to test_questions.json if input doesn't exist
        alt_path = BACKEND_DIR / "test_questions.json"
        if alt_path.exists():
            input_path = alt_path
        else:
            raise FileNotFoundError(f"Input file not found at {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:  # noqa: ASYNC230
        questions = json.load(f)

    if args.limit:
        questions = questions[: args.limit]

    print(f"Loaded {len(questions)} questions from {input_path}")
    print(f"Using provider: {args.provider}, k: {args.k}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, item in enumerate(questions, start=1):
        q = item.get("question", "")
        print(f"[{i}/{len(questions)}] Processing: {q[:60]}...")
        eval_result = await run_single_eval(item, provider=args.provider, k=args.k)
        results.append(eval_result)

        # Save incrementally after each question so progress is never lost
        with open(output_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully evaluated {len(results)} questions.")
    print(f"Results saved to: {output_path}")

    # Also mirror to root directory if run inside backend
    root_output = BACKEND_DIR.parent / "rag_tests_questions.json"
    if root_output.parent.exists() and root_output.parent != BACKEND_DIR:
        with open(root_output, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
