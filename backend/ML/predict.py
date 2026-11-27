import asyncio
from pathlib import Path

import joblib

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

# ==========================================
# LOAD SAVED MODELS
# ==========================================

def _find_file(filename: str):
    candidates = [
        CURRENT_DIR / filename,
        BASE_DIR / filename,
        Path(filename),
        CURRENT_DIR / "saved_models" / filename,
        BASE_DIR / "saved_models" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def _load_artifacts():
    svm_path = _find_file("linear_svm_model.pkl") or _find_file("best_security_model.pkl")
    nb_path = _find_file("naive_bayes_model.pkl")
    vec_path = _find_file("tfidf_vectorizer.pkl")
    bundle_path = _find_file("prompt_injection_detector.pkl")

    svm = joblib.load(svm_path) if svm_path else None
    nb = joblib.load(nb_path) if nb_path else None
    vec = joblib.load(vec_path) if vec_path else None

    if (svm is None or nb is None or vec is None) and bundle_path:
        bundle = joblib.load(bundle_path)
        if svm is None:
            svm = bundle.get("svm")
        if nb is None:
            nb = bundle.get("naive_bayes")
        if vec is None:
            vec = bundle.get("vectorizer")

    return svm, nb, vec

svm_model, nb_model, vectorizer = _load_artifacts()


# ==========================================
# 1. LINEAR SVM DETECTION
# ==========================================

async def detect_with_svm(text: str):
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    global svm_model, vectorizer
    if svm_model is None or vectorizer is None:
        svm_model, _, vectorizer = _load_artifacts()

    # Convert text into TF-IDF features
    text_vector = vectorizer.transform([text])

    # Predict
    prediction = svm_model.predict(text_vector)[0]

    return {
        "model": "Linear SVM",
        "prediction": int(prediction),
        "label": "jailbreak" if prediction == 1 else "benign"
    }


# ==========================================
# 2. NAIVE BAYES DETECTION
# ==========================================

async def detect_with_naive_bayes(text: str):
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    global nb_model, vectorizer
    if nb_model is None or vectorizer is None:
        _, nb_model, vectorizer = _load_artifacts()

    # Convert text into TF-IDF features
    text_vector = vectorizer.transform([text])

    # Predict
    prediction = nb_model.predict(text_vector)[0]

    return {
        "model": "Naive Bayes",
        "prediction": int(prediction),
        "label": "jailbreak" if prediction == 1 else "benign"
    }


# ==========================================
# 3. COMBINED DETECTION
# ==========================================

async def detect_with_both(text: str):
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    # Run both models simultaneously
    svm_result, nb_result = await asyncio.gather(
        detect_with_svm(text),
        detect_with_naive_bayes(text)
    )

    svm_prediction = svm_result["prediction"]
    nb_prediction = nb_result["prediction"]

    # Majority / either detection logic
    final_prediction = 1 if (svm_prediction == 1 or nb_prediction == 1) else 0

    return {
        "query": text,
        "svm": svm_result["label"],
        "naive_bayes": nb_result["label"],
        "final_prediction": final_prediction,
        "final_label": "jailbreak" if final_prediction == 1 else "benign",
        "both_agree": svm_result["label"] == nb_result["label"]
    }


async def main():
    text = "what is corruption"
    r = await detect_with_both(text=text)
    print("Result:", r)

if __name__ == "__main__":
    asyncio.run(main())
