import asyncio

import joblib

# ==========================================
# LOAD SAVED MODELS
# ==========================================

svm_model = joblib.load("linear_svm_model.pkl")
nb_model = joblib.load("naive_bayes_model.pkl")

vectorizer = joblib.load("tfidf_vectorizer.pkl")


# ==========================================
# 1. LINEAR SVM DETECTION
# ==========================================

async def detect_with_svm(text: str):

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

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
# TESTING
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

    # Majority voting (two-model agreement)
    final_prediction = (
        1 if svm_prediction == 1 or nb_prediction == 1 else 0
    )

    return {
        "query": text,
        "svm": svm_result["label"],
        "naive_bayes": nb_result["label"],
        "final_prediction": final_prediction,
        "final_label": (
            "jailbreak" if final_prediction == 1 else "benign"
        ),
        "both_agree" : svm_result['label'] == nb_result['label']
    }


async def main():

    text = "what is corruption"

    r = await detect_with_both(text=text)

    print(r)


if __name__ == "__main__":
    asyncio.run(main())
