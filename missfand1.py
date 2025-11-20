import pandas as pd

# 全部打印，不省略行列（如果太多可以自己改回默认）
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ① 读取四个模型的 LOOCV 偽陰性データ
dt_fn  = pd.read_csv("LOOCV_FN_Tree.csv")   # 決定木
rf_fn  = pd.read_csv("LOOCV_FN_Rf.csv")     # ランダムフォレスト
log_fn = pd.read_csv("LOOCV_FN_Lr.csv")     # ロジスティック回帰
svm_fn = pd.read_csv("LOOCV_FN_Svm.csv")    # SVM

# ② 用于识别“同一个人”的特征列（你给的那一套）
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

# ③ 定义一个小函数：去重 + 只保留键 + 标记这个模型的FN=1
def prepare_fn(df, flag_name):
    # 按 key_cols 去重，避免一个人出现多行
    df_u = df.drop_duplicates(subset=key_cols).copy()
    df_u[flag_name] = 1
    # 只保留 键 + 标记列，避免 merge 时列名冲突
    return df_u[key_cols + [flag_name]]

dt_p  = prepare_fn(dt_fn,  "FN_Tree")
rf_p  = prepare_fn(rf_fn,  "FN_RF")
log_p = prepare_fn(log_fn, "FN_LR")
svm_p = prepare_fn(svm_fn, "FN_SVM")

# ④ 依次外连接，把所有在任一模型中被判为FN的人都合到一张表里
merged = (
    dt_p
    .merge(rf_p,  on=key_cols, how="outer")
    .merge(log_p, on=key_cols, how="outer")
    .merge(svm_p, on=key_cols, how="outer")
)

# ⑤ 没有出现在某个模型FN列表的人，标记为 0
for col in ["FN_Tree", "FN_RF", "FN_LR", "FN_SVM"]:
    merged[col] = merged[col].fillna(0).astype(int)

# ⑥ 统计每个人被多少个模型共同判错
merged["FN_count"] = merged[["FN_Tree", "FN_RF", "FN_LR", "FN_SVM"]].sum(axis=1)

print("\n=== 每个人被多少个模型判错的分布（FN_count） ===")
print(merged["FN_count"].value_counts().sort_index())

# ⑦ 保存成 CSV，方便在 Excel 里看
merged.to_csv("miss1.csv", index=False, encoding="utf-8-sig")
print("\nmiss1.csv")

# ⑧ 在终端逐条显示：这个人被哪些模型判错 + 主要特征
print("\n=== 逐条列出每个被判错的个体及其对应的模型 ===\n")

for idx, row in merged.iterrows():
    models = []
    if row["FN_Tree"] == 1:
        models.append("決定木")
    if row["FN_RF"] == 1:
        models.append("ランダムフォレスト")
    if row["FN_LR"] == 1:
        models.append("ロジスティック回帰")
    if row["FN_SVM"] == 1:
        models.append("SVM")

    print("======================================")
    print(f"個体 {idx+1} / FN_count = {row['FN_count']}")
    print("判定ミスしたモデル:", "・".join(models))
    print("特徴：")
    for k in key_cols:
        print(f" - {k}: {row[k]}")