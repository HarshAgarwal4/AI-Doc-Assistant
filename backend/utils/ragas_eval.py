import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Ensure backend directory is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

from datasets import Dataset
from langchain_ollama import OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from utils.llm import get_llm
from utils.logger import logger


def load_evaluation_data(file_path: Path) -> list[dict]:
    if not file_path.exists():
        alt_path = BACKEND_DIR / "rag_testes_questions.json"
        if alt_path.exists():
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot find input dataset at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def format_table_cell(text: str) -> str:
    """Format string for clean markdown table rendering."""
    if not text:
        return ""
    # Replace newlines with <br> and escape markdown pipes
    cleaned = str(text).replace("\r\n", "\n").replace("\n", "<br>").replace("|", "\\|")
    return cleaned.strip()


def format_score(score: float | None, precision: int = 4) -> str:
    """Format metric score for display; returns 'N/A' if None or NaN."""
    if score is None:
        return "N/A"
    try:
        val = float(score)
        if math.isnan(val):
            return "N/A"
        return f"{val:.{precision}f}"
    except (ValueError, TypeError):
        return "N/A"


def build_markdown_report(
    results_data: list[dict],
    summary_metrics: dict,
    output_file: Path,
    failed_data: list[dict] | None = None,
) -> str:
    avg_score_str = format_score(summary_metrics.get("average"))
    failed_count = len(failed_data) if failed_data else 0
    if failed_count > 0:
        eval_summary_line = (
            f"**Overall Average Score:** **{avg_score_str}** "
            f"(Evaluated over {len(results_data)} valid generation samples; {failed_count} failed generation samples excluded)"
        )
    else:
        eval_summary_line = (
            f"**Overall Average Score:** **{avg_score_str}** "
            f"(Evaluated over {len(results_data)} test questions; failed metrics excluded)"
        )

    md_lines = [
        "# RAGAS Evaluation Report",
        "",
        "Comprehensive evaluation of the RAG pipeline using RAGAS metrics over 25 ground-truth document questions from the **Helios Meridian Technologies FY2025 Annual Technical & Operations Report**.",
        "",
        "## 📊 Executive Summary & Aggregate Metrics",
        "",
        "| Metric | Score | Description |",
        "|---|:---:|---|",
        f"| **Faithfulness** | **{format_score(summary_metrics.get('faithfulness'))}** | Factual consistency of the generated answer grounded in the retrieved context |",
        f"| **Answer Relevancy** | **{format_score(summary_metrics.get('answer_relevancy'))}** | Direct pertinence of the answer to the user query |",
        f"| **Context Precision** | **{format_score(summary_metrics.get('context_precision'))}** | Signal-to-noise ratio and ranking quality of retrieved document chunks |",
        f"| **Context Recall** | **{format_score(summary_metrics.get('context_recall'))}** | Coverage of the expected ground truth contained within the retrieved context |",
        "",
        eval_summary_line,
        "",
        "---",
        "",
        "## 📑 Detailed Question-by-Question Evaluation Table",
        "",
        "| # | Question | Ground Truth | Generated Answer | Retrieved Contexts | Faithfulness | Answer Relevancy | Context Precision | Context Recall |",
        "|---|---|---|---|---|:---:|:---:|:---:|:---:|",
    ]

    for idx, item in enumerate(results_data, start=1):
        q = format_table_cell(item.get("question", ""))
        gt = format_table_cell(item.get("ground_truth") or item.get("grounded_truth", ""))
        ans = format_table_cell(item.get("answer", ""))

        ctx_list = item.get("contexts", [])
        if isinstance(ctx_list, list):
            formatted_ctx = "<br>• ".join([format_table_cell(c) for c in ctx_list])
            if formatted_ctx:
                formatted_ctx = "• " + formatted_ctx
        else:
            formatted_ctx = format_table_cell(str(ctx_list))

        f_score = format_score(item.get("faithfulness"), precision=2)
        ar_score = format_score(item.get("answer_relevancy"), precision=2)
        cp_score = format_score(item.get("context_precision"), precision=2)
        cr_score = format_score(item.get("context_recall"), precision=2)

        row = (
            f"| {idx} | {q} | {gt} | {ans} | {formatted_ctx} "
            f"| {f_score} | {ar_score} | {cp_score} | {cr_score} |"
        )
        md_lines.append(row)

    if failed_data:
        md_lines.extend([
            "",
            "---",
            "",
            "## ⚠️ Excluded Generation Failures",
            "",
            "The following samples encountered generation failures (e.g. API timeouts, empty responses, or provider errors) and were explicitly excluded from RAGAS benchmark metrics to prevent distorting model performance:",
            "",
            "| # | Question | Error Reason | Status |",
            "|---|---|---|:---:|",
        ])
        for f_idx, item in enumerate(failed_data, start=1):
            f_q = format_table_cell(item.get("question", ""))
            f_err = format_table_cell(
                item.get("error")
                or item.get("retrieval_error")
                or "Generation failure / empty answer"
            )
            md_lines.append(f"| {f_idx} | {f_q} | {f_err} | ❌ Failed |")

    content = "\n".join(md_lines) + "\n"

    # Save report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    # Mirror to root directory if in backend
    root_file = BACKEND_DIR.parent / "RAGAS.md"
    if root_file.parent.exists() and root_file.parent != BACKEND_DIR:
        with open(root_file, "w", encoding="utf-8") as f:
            f.write(content)

    return content


def run_evaluation(
    input_file: Path = BACKEND_DIR / "rag_testes_questions.json",
    output_file: Path = BACKEND_DIR / "RAGAS.md",
    provider: str = "Ollama",
    raise_exceptions: bool = False,
):
    print(f"Loading test questions from: {input_file}")
    raw_data = load_evaluation_data(input_file)
    print(f"Loaded {len(raw_data)} questions.")

    # Explicitly separate valid samples from failed generation samples
    valid_data = [
        item for item in raw_data
        if item.get("status", "success") == "success" and bool(str(item.get("answer", "")).strip())
    ]
    failed_data = [item for item in raw_data if item not in valid_data]

    if failed_data:
        print(f"Detected {len(failed_data)} failed generation sample(s) out of {len(raw_data)} total.")
        print(f"Excluding {len(failed_data)} failed sample(s) from RAGAS evaluation to prevent metric distortion.")

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    if not valid_data:
        print("No valid generation samples available for evaluation.")
        summary_metrics = {m: None for m in metric_names}
        summary_metrics["average"] = None
        build_markdown_report([], summary_metrics, output_file, failed_data=failed_data)
        return

    questions = [item.get("question", "") for item in valid_data]
    answers = [item.get("answer", "") for item in valid_data]
    contexts = [item.get("contexts", []) for item in valid_data]
    ground_truths = [item.get("ground_truth") or item.get("grounded_truth", "") for item in valid_data]

    hf_dataset = Dataset.from_dict({
        "question": questions,
        "user_input": questions,
        "answer": answers,
        "response": answers,
        "contexts": contexts,
        "retrieved_contexts": contexts,
        "ground_truth": ground_truths,
        "reference": ground_truths,
    })

    # Initialize Evaluator LLM and Embeddings
    try:
        eval_llm = get_llm(provider)
        ragas_llm = LangchainLLMWrapper(eval_llm)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not initialize {provider} LLM, falling back: {e}")
        eval_llm = get_llm("Ollama")
        ragas_llm = LangchainLLMWrapper(eval_llm)

    try:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ragas_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_host)
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not initialize embeddings: {e}")
        ragas_embeddings = None

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    print("Running RAGAS evaluation across all 4 metrics...")
    try:
        results = evaluate(
            dataset=hf_dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=raise_exceptions,
            show_progress=True,
        )
        print("Evaluation completed successfully.")

        # Extract per-question metrics (strictly avoid fabricating scores)
        df = results.to_pandas()
        results_data = []
        for i, item in enumerate(valid_data):
            row_dict = dict(item)
            for m in metric_names:
                if m in df.columns:
                    val = df[m].iloc[i]
                    if val is not None and not pd.isna(val):
                        float_val = float(val)
                        row_dict[m] = None if math.isnan(float_val) else float_val
                    else:
                        row_dict[m] = None
                else:
                    row_dict[m] = None
            results_data.append(row_dict)

        summary_metrics = {}
        valid_metric_means = []
        for m in metric_names:
            valid_vals = [
                r[m] for r in results_data
                if r.get(m) is not None and not math.isnan(r[m])
            ]
            if valid_vals:
                m_mean = sum(valid_vals) / len(valid_vals)
                summary_metrics[m] = m_mean
                valid_metric_means.append(m_mean)
            else:
                summary_metrics[m] = None

        # Failed metric values should never contribute to the aggregate average
        if valid_metric_means:
            summary_metrics["average"] = sum(valid_metric_means) / len(valid_metric_means)
        else:
            summary_metrics["average"] = None

    except Exception as e:
        if raise_exceptions:
            raise
        logger.error(f"RAGAS evaluation failed: {e}")
        print(f"RAGAS evaluation failed: {e}. Marking all metrics as unavailable (None/NaN).")

        results_data = []
        for item in valid_data:
            entry = dict(item)
            for m in metric_names:
                entry[m] = None
            results_data.append(entry)

        summary_metrics = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
            "average": None,
        }

    print("Building Markdown report...")
    build_markdown_report(results_data, summary_metrics, output_file, failed_data=failed_data)
    print(f"Report generated successfully at: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on rag_testes_questions.json")
    parser.add_argument(
        "--input",
        type=str,
        default=str(BACKEND_DIR / "rag_testes_questions.json"),
        help="Path to input test questions file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BACKEND_DIR / "RAGAS.md"),
        help="Path to output markdown report",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="Ollama",
        choices=["Groq", "Google", "Mistral", "Ollama"],
        help="Provider for Ragas evaluation LLM",
    )
    parser.add_argument(
        "--raise-exceptions",
        action="store_true",
        default=False,
        help="Propagate evaluation exceptions instead of catching and marking metrics as unavailable",
    )
    args = parser.parse_args()

    run_evaluation(
        input_file=Path(args.input),
        output_file=Path(args.output),
        provider=args.provider,
        raise_exceptions=args.raise_exceptions,
    )

