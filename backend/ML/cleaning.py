from pathlib import Path

import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

csv_candidates = [
    CURRENT_DIR / "prompt_injection.csv",
    BASE_DIR / "prompt_injection.csv",
    Path("prompt_injection.csv"),
    Path("d:/Training/ML/prompt injection/prompt_injection.csv"),
]

csv_path = None
for p in csv_candidates:
    if p.exists():
        csv_path = p
        break

if not csv_path:
    raise FileNotFoundError("Could not find prompt_injection.csv")

print(f"Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

print("\n========== INITIAL DATASET ==========")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nDataset Information:")
df.info()


# ==========================================
# 2. CLEAN COLUMN NAMES
# ==========================================

df.columns = df.columns.str.strip().str.lower()

# Fix incorrectly formatted column names if present
df = df.rename(columns={
    "**source": "source",
    "**&#x73;plit": "split",
})

print("\nCleaned Columns:")
print(df.columns.tolist())


# ==========================================
# 3. REMOVE DUPLICATES
# ==========================================

print("\n========== DUPLICATE CHECK ==========")
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

if "text" in df.columns:
    print("Duplicate text records:", df.duplicated(subset=["text"]).sum())

if "text" in df.columns and "category" in df.columns:
    conflicts = df.groupby("text")["category"].nunique()
    print("Conflicting text categories:", (conflicts > 1).sum())


# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Text is essential
if "text" in df.columns:
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

# Category handling
if "category" in df.columns:
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["category"] != ""]


# ==========================================
# 5. GENERATE BINARY LABELS
# ==========================================

print("\n========== LABEL GENERATION ==========")

# Benign = 0, Attack = 1
if "category" in df.columns:
    df["label"] = df["category"].str.lower().apply(lambda x: 0 if x == "benign" else 1)
elif "label" in df.columns:
    if df["label"].dtype == "object":
        df["label"] = df["label"].str.lower().apply(lambda x: 0 if x == "benign" else 1)
    else:
        df["label"] = df["label"].astype(int)

print("Label Distribution:")
print(df["label"].value_counts())

print("\nLabel Percentages:")
print(df["label"].value_counts(normalize=True) * 100)

if "category" in df.columns:
    print("\nCategory-wise Distribution:")
    print(df["category"].value_counts())


# ==========================================
# 6. CLEAN TEXT
# ==========================================

print("\n========== TEXT CLEANING ==========")

# Normalize whitespace
df["text"] = df["text"].str.strip()
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)

# Recalculate word count
df["word_count"] = df["text"].str.split().str.len()
print("Word count updated successfully.")


# ==========================================
# 7. CHECK TRAIN/TEST SPLIT
# ==========================================

if "split" in df.columns:
    print("\n========== TRAIN TEST SPLIT ==========")
    print(df["split"].value_counts(dropna=False))
    print("\nSplit-wise Label Distribution:")
    print(pd.crosstab(df["split"], df["label"], margins=True))


# ==========================================
# 8. CHECK ID AND DATA CONSISTENCY
# ==========================================

print("\n========== DATA VALIDATION ==========")

if "id" in df.columns:
    print("Duplicate IDs:", df["id"].duplicated().sum())

if "language" in df.columns:
    print("\nLanguage Distribution:")
    print(df["language"].value_counts(dropna=False))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFinal Label Distribution:")
print(df["label"].value_counts())


# ==========================================
# 9. FINAL VALIDATION
# ==========================================

assert df["text"].notna().all(), "Missing text found!"
assert df["label"].notna().all(), "Missing labels found!"
assert set(df["label"].unique()) == {0, 1}, "Both benign and malicious classes must exist!"

print("\nDataset validation successful!")


# ==========================================
# 10. SAVE CLEANED DATASET
# ==========================================

out_path = BASE_DIR / "clean_prompt_injection.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print("\n========== FINAL RESULT ==========")
print(f"Cleaned dataset saved successfully to {out_path.name}!")
print("Final dataset shape:", df.shape)

print("\nFinal Dataset Preview:")
print(df.head())
