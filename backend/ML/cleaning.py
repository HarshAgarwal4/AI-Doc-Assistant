
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

jailbreak = pd.concat(
    [jailbreak_train, jailbreak_test],
    ignore_index=True
)

jailbreak = jailbreak[["prompt", "type"]]

jailbreak.columns = ["text", "label"]

jailbreak["label"] = jailbreak["label"].str.lower().map({
    "benign": 0,
    "jailbreak": 1
})

jailbreak["source"] = "jailbreak"


# ==========================================
# 2. SALAD ATTACK-ENHANCED DATASET
# ==========================================

salad_url = (
    "https://huggingface.co/datasets/"
    "OpenSafetyLab/Salad-Data/resolve/main/"
    "attack_enhanced_set.json"
)

salad = pd.read_json(salad_url)

salad = salad[["augq"]]

salad.columns = ["text"]

# Weak positive labels:
# These are attack-enhanced safety prompts,
# not all necessarily prompt injections.
salad["label"] = 1

salad["source"] = "salad"
# ==========================================
# 3. COMBINE DATASETS
# ==========================================

df = pd.concat(
    [jailbreak, salad ],
    ignore_index=True
)

# Remove empty text
df["text"] = df["text"].astype("string").str.strip()

df = df.dropna(subset=["text", "label"])

df = df[df["text"] != ""]

# Remove exact duplicate texts
df = df.drop_duplicates(subset=["text"])

# Keep valid labels
df["label"] = df["label"].astype(int)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.drop(columns=['source'])

# ==========================================
# 4. SAVE FINAL DATASET
# ==========================================

df.to_csv("rag_security_training.csv", index=False)

print("\nDataset created successfully!")

print("\nTotal examples:", len(df))

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nSample data:")
print(df.head())