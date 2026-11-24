import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# ==========================================
# 1. LOAD PRESERVED TRAIN AND TEST DATASETS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

train_path = Path("rag_security_training.csv")
if not train_path.exists():
    train_path = BASE_DIR / "rag_security_training.csv"

test_path = Path("rag_security_test.csv")
if not test_path.exists():
    test_path = BASE_DIR / "rag_security_test.csv"

train_df = pd.read_csv(train_path)
train_df = train_df.dropna(subset=["text", "label"])
train_df["text"] = train_df["text"].astype(str).str.strip()
train_df = train_df[train_df["text"] != ""]
train_df["label"] = train_df["label"].astype(int)
train_df = train_df.drop_duplicates(subset=["text"])

test_df = pd.read_csv(test_path)
test_df = test_df.dropna(subset=["text", "label"])
test_df["text"] = test_df["text"].astype(str).str.strip()
test_df = test_df[test_df["text"] != ""]
test_df["label"] = test_df["label"].astype(int)
test_df = test_df.drop_duplicates(subset=["text"])

# Preserve independent test-set guarantee: strictly purge any test samples from training data
train_df = train_df[~train_df["text"].isin(test_df["text"])].reset_index(drop=True)

X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]

print(f"Preserved independent split: {len(X_train)} train samples, {len(X_test)} untouched test samples.")
print("\nTrain label distribution:")
print(y_train.value_counts())
print("\nUntouched Test label distribution:")
print(y_test.value_counts())


# ==========================================
# 2. TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=10000
)

# Fit vectorizer strictly on training data
X_train_vec = vectorizer.fit_transform(X_train)
# Transform untouched test data
X_test_vec = vectorizer.transform(X_test)

print("\nVocabulary size:", len(vectorizer.vocabulary_))


# ==========================================
# 3. INITIALIZE MODELS
# ==========================================

models = {
    "Linear SVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42
    ),

    "Naive Bayes": MultinomialNB(
        alpha=0.1
    )
}


# ==========================================
# 4. TRAIN ON TRAIN SET & EVALUATE ON UNTOUCHED TEST SET
# ==========================================

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name} on training dataset...")

    start_time = time.time()
    model.fit(X_train_vec, y_train)
    training_time = time.time() - start_time

    # Evaluate strictly on untouched test dataset
    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time (sec)": round(training_time, 4)
    })

    trained_models[name] = model

    print(f"\n{name} Classification Report (Untouched Test Dataset):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Jailbreak"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ==========================================
# 5. COMPARE MODELS
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ON UNTOUCHED TEST DATASET ==========")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)


# ==========================================
# 6. SELECT BEST MODEL
# ==========================================

best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]
print("\nBest Model on Untouched Test Set:", best_model_name)


# ==========================================
# 7. SAVE MODELS & ARTIFACTS
# ==========================================

save_dir = BASE_DIR
joblib.dump(best_model, save_dir / "best_security_model.pkl")
joblib.dump(vectorizer, save_dir / "tfidf_vectorizer.pkl")
joblib.dump(trained_models["Linear SVM"], save_dir / "linear_svm_model.pkl")
joblib.dump(trained_models["Naive Bayes"], save_dir / "naive_bayes_model.pkl")

# Also dump to current directory if running inside backend/ML
if Path.cwd() != save_dir:
    joblib.dump(best_model, "best_security_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(trained_models["Linear SVM"], "linear_svm_model.pkl")
    joblib.dump(trained_models["Naive Bayes"], "naive_bayes_model.pkl")

print("\nAll models and vectorizer trained and saved successfully with independent test-set guarantee!")
