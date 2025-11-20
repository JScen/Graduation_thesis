import pandas as pd

# 读取 clean.csv
df = pd.read_csv("clean.csv")

# 你拥有的全部特征（除了目标变量）
key_cols = [
    "入国時年齢",
    "性別",
    "職種関係",
    "職種",
    "所在地(実習先)(都道府県)",
    "所在地(実習先)(市区町村)",
    "派遣会社",
    "学校所属",
    "組合",
    "所属機関"
]

# Step 1：按组合分组，查看每组的标签数量（0/1）
conflict_groups = (
    df.groupby(key_cols)["失踪の有無"]
      .nunique()
      .reset_index(name="label種類数")
)

# Step 2：筛选出“同组内有0也有1”的矛盾组
conflict_groups = conflict_groups[conflict_groups["label種類数"] > 1]

print("\n=== 矛盾组合数量（同一组内既有失踪又有未失踪）===\n")
print(len(conflict_groups))

print("\n=== 矛盾组合（特征组合）列表 ===\n")
print(conflict_groups[key_cols].to_string(index=False))

# Step 3：导出这些组合具体有哪些人
conflict_rows = df.merge(conflict_groups[key_cols], on=key_cols, how="inner")

print("\n=== 矛盾组的全部个人记录（含失踪の有無） ===\n")
print(conflict_rows[key_cols + ["失踪の有無"]].to_string(index=False))

# 保存结果
conflict_rows.to_csv("same people.csv", index=False, encoding="utf-8-sig")
print("\n same people.csv")