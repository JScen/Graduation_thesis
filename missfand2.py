import pandas as pd

# 读四个文件（原始 FN 文件）
dt_fn  = pd.read_csv("LOOCV_FN_Tree.csv")
rf_fn  = pd.read_csv("LOOCV_FN_Rf.csv")
log_fn = pd.read_csv("LOOCV_FN_Lr.csv")
svm_fn = pd.read_csv("LOOCV_FN_Svm.csv")

# 要检查的特征列
cols_to_check = [
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

def show_fn_dist(df, model_name):
    print(f"\n==============================")
    print(f"{model_name}：偽陰性データの特征分布")
    print(f"==============================\n")

    # 存储所有分布到一个 dict
    dist_dict = {}

    for col in cols_to_check:
        if col in df.columns:
            print(f"\n--- {col} ---")
            vc = df[col].value_counts(dropna=False)
            print(vc.to_string())

            # 保存到字典，以便写入 CSV
            dist_dict[col] = vc

    # ==== 新增：保存到 CSV ====
    output_rows = []
    for col, vc in dist_dict.items():
        for value, count in vc.items():
            output_rows.append([col, value, count])

    # 生成 DataFrame
    out_df = pd.DataFrame(output_rows,
                          columns=["特徴名", "値", "数"])

    # 文件名：FN_dist_模型.csv
    filename = f"miss2.{model_name}.csv"
    out_df.to_csv(filename, index=False, encoding="utf-8-sig")

    print(f"\n{filename}\n")


# 各模型分别输出 + 保存
show_fn_dist(dt_fn,  "DecisionTree")
show_fn_dist(rf_fn,  "RandomForest")
show_fn_dist(log_fn, "LogisticRegression")
show_fn_dist(svm_fn, "SVM")