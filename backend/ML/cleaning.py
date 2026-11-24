
import pandas as pd

# ==========================================
# 1. JAILBREAK CLASSIFICATION DATASET
# ==========================================

train_url = (
    "https://huggingface.co/datasets/"
    "jackhhao/jailbreak-classification/resolve/main/"
    "balanced/jailbreak_dataset_train_balanced.csv"
)

test_url = (
    "https://huggingface.co/datasets/"
    "jackhhao/jailbreak-classification/resolve/main/"
    "balanced/jailbreak_dataset_test_balanced.csv"
)

jailbreak_train = pd.read_csv(train_url)
jailbreak_test = pd.read_csv(test_url)

# PRESERVE SUPPLIED SPLIT: Keep train and test datasets independent
jailbreak_train = jailbreak_train[["prompt", "type"]].rename(columns={"prompt": "text", "type": "label"})
jailbreak_train["label"] = jailbreak_train["label"].str.lower().map({
    "benign": 0,
    "jailbreak": 1
})

jailbreak_test = jailbreak_test[["prompt", "type"]].rename(columns={"prompt": "text", "type": "label"})
jailbreak_test["label"] = jailbreak_test["label"].str.lower().map({
    "benign": 0,
    "jailbreak": 1
})

# ==========================================
# 2. SALAD ATTACK-ENHANCED DATASET (TRAINING AUGMENTATION)
# ==========================================

salad_url = (
    "https://huggingface.co/datasets/"
    "OpenSafetyLab/Salad-Data/resolve/main/"
    "attack_enhanced_set.json"
)

salad = pd.read_json(salad_url)
salad = salad[["augq"]].rename(columns={"augq": "text"})
salad["label"] = 1

# ==========================================
# 3. SAVE UNTOUCHED TEST DATASET
# ==========================================

test_df = jailbreak_test.dropna(subset=["text", "label"]).copy()
test_df["text"] = test_df["text"].astype("string").str.strip()
test_df = test_df[test_df["text"] != ""]
test_df = test_df.drop_duplicates(subset=["text"])
test_df["label"] = test_df["label"].astype(int)
test_df.to_csv("rag_security_test.csv", index=False)
print("Saved untouched test dataset: rag_security_test.csv with shape", test_df.shape)

# ==========================================
# 4. COMBINE AND CLEAN TRAINING DATASET ONLY
# ==========================================

train_df = pd.concat([jailbreak_train, salad], ignore_index=True)
train_df["text"] = train_df["text"].astype("string").str.strip()
train_df = train_df.dropna(subset=["text", "label"])
train_df = train_df[train_df["text"] != ""]
train_df = train_df.drop_duplicates(subset=["text"])
train_df["label"] = train_df["label"].astype(int)

# Independent test-set guarantee: remove any test samples from training data
train_df = train_df[~train_df["text"].isin(test_df["text"])].reset_index(drop=True)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df.to_csv("rag_security_training.csv", index=False)
print("Saved training dataset: rag_security_training.csv with shape", train_df.shape)

print("\nTrain dataset created successfully!")
print("Total train examples:", len(train_df))
print("Total test examples:", len(test_df))
print("\nTrain label distribution:")
print(train_df["label"].value_counts())
print("\nTest label distribution:")
print(test_df["label"].value_counts())
