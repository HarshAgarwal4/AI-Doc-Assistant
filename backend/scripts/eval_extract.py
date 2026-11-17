import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:5000"
EXTRACT_URL = f"{BASE_URL}/extract"

RESUME_DIR = Path("resumes")
EXPECTED_FILE = Path("expected_outputs.json")

DELAY = 3  # Seconds between API calls

FIELDS = ["name", "email", "phone", "skills"]


def load_expected_outputs():
    with open(EXPECTED_FILE, encoding="utf-8") as file:
        return json.load(file)


def evaluate_resume(resume_path, expected):
    start_time = time.perf_counter()

    with open(resume_path, "rb") as file:
        response = requests.post(
            EXTRACT_URL,
            files={
                "file": (
                    resume_path.name,
                    file,
                    "application/pdf",
                )
            },
            timeout=120,
        )

    api_time = time.perf_counter() - start_time

    if response.status_code != 200:
        logger.error(
            "[FAILED] %s: HTTP %s | Time: %.2fs",
            resume_path.name,
            response.status_code,
            api_time,
        )
        return None

    actual = response.json()

    correct = 0
    total = len(FIELDS)

    for field in FIELDS:
        if actual.get(field) == expected.get(field):
            correct += 1
        else:
            logger.warning(
                "  Incorrect %s: Expected=%s, Actual=%s",
                field,
                expected.get(field),
                actual.get(field),
            )

    logger.info("  API call time: %.2f seconds", api_time)

    return {
        "valid": True,
        "correct": correct,
        "total": total,
        "api_time": api_time,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    total_start_time = time.perf_counter()

    expected_outputs = load_expected_outputs()

    total_resumes = len(expected_outputs)
    valid_resumes = 0
    correct_fields = 0
    total_fields = 0
    total_api_time = 0.0
    successful_calls = 0

    logger.info("Evaluating %d resumes...", total_resumes)

    for index, expected in enumerate(expected_outputs):
        filename = expected["filename"]
        resume_path = RESUME_DIR / filename

        logger.info("[%d/%d] %s", index + 1, total_resumes, filename)

        if not resume_path.exists():
            logger.warning("  [MISSING] Resume file not found")
            continue

        try:
            result = evaluate_resume(
                resume_path,
                expected["expected_output"],
            )

            if result is not None:
                valid_resumes += 1
                correct_fields += result["correct"]
                total_fields += result["total"]
                total_api_time += result["api_time"]
                successful_calls += 1

                logger.info(
                    "  Correct fields: %d/%d",
                    result["correct"],
                    result["total"],
                )

        except requests.RequestException as error:
            logger.error("  [ERROR] API request failed: %s", error)

        except (ValueError, KeyError) as error:
            logger.error("  [ERROR] Invalid response: %s", error)

        if index < total_resumes - 1:
            time.sleep(DELAY)

    total_execution_time = time.perf_counter() - total_start_time

    accuracy = (
        correct_fields / total_fields * 100
        if total_fields else 0
    )

    average_api_time = (
        total_api_time / successful_calls
        if successful_calls else 0
    )

    logger.info("========== EVALUATION SUMMARY ==========")
    logger.info("Total resumes:          %d", total_resumes)
    logger.info("Valid responses:        %d", valid_resumes)
    logger.info("Failed responses:       %d", total_resumes - valid_resumes)
    logger.info("Correct fields:         %d", correct_fields)
    logger.info("Total fields:           %d", total_fields)
    logger.info("Field accuracy:         %.2f%%", accuracy)
    logger.info("----------------------------------------")
    logger.info("Total API time:         %.2f seconds", total_api_time)
    logger.info("Average API time:       %.2f seconds", average_api_time)
    logger.info("Total execution time:   %.2f seconds", total_execution_time)
    logger.info("========================================")


if __name__ == "__main__":
    main()