import pandas as pd

df = pd.read_csv("clean.csv")

# 仅查看类别型字段（object、string类型）
cat_cols = df.select_dtypes(include=["object", "string"]).columns

for col in cat_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts().head(10))
    print(f"合計カテゴリ数: {df[col].nunique()}")

counts = {}

for col in cat_cols:
    counts[col] = df[col].value_counts()

# 各列をDataFrameにまとめる
counts_df = pd.concat(counts, axis=1)
counts_df.to_csv("kazu.csv", encoding="utf-8-sig")