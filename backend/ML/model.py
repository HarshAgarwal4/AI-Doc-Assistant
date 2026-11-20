import time

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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("rag_security_training.csv")

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df["label"] = df["label"].astype(int)
df = df.drop_duplicates(subset=["text"])

X = df["text"]
y = df["label"]

print("Dataset shape:", df.shape)
print("\nLabel distribution:")
print(y.value_counts())


# ==========================================
# 2. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=10000
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("\nVocabulary size:", len(vectorizer.vocabulary_))


# ==========================================
# 4. INITIALIZE MODELS
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
# 5. TRAIN AND EVALUATE
# ==========================================

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time (sec)": round(training_time, 4)
    })

    trained_models[name] = model

    print(f"\n{name} Classification Report:")

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
# 6. COMPARE MODELS
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)


# ==========================================
# 7. SELECT BEST MODEL
# ==========================================

best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)


# ==========================================
# 8. SAVE MODELS
# ==========================================

joblib.dump(best_model, "best_security_model.pkl")

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

joblib.dump(
    trained_models["Linear SVM"],
    "linear_svm_model.pkl"
)

joblib.dump(
    trained_models["Naive Bayes"],
    "naive_bayes_model.pkl"
)

print("\nAll models and vectorizer saved successfully!")