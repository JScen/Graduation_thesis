import pandas as pd
from pathlib import Path

dt_fn  = pd.read_csv("LOOCV_FN_Tree.csv")
rf_fn  = pd.read_csv("LOOCV_FN_Rf.csv")
log_fn = pd.read_csv("LOOCV_FN_Lr.csv")
svm_fn = pd.read_csv("LOOCV_FN_Svm.csv")

KEY_COLS_ALL = [
    "組合", "企業名", "期数",
    "所在地(実習先)(都道府県)", "所在地(実習先)(市区町村)",
    "派遣会社", "所属機関", "性別", "入国時年齢",
    "学校所属", "職種関係", "職種", "作業",
    "入国日", "失踪日", "失踪までの在日日数"
]

def common_key_cols(dfs, cand_cols):
    cols = set(cand_cols)
    for df in dfs:
        cols &= set(df.columns)
    return [c for c in cand_cols if c in cols]

KEY_COLS = common_key_cols([dt_fn, rf_fn, log_fn, svm_fn], KEY_COLS_ALL)
print(KEY_COLS)

def build_key(df, key_cols):
    sub = df[key_cols].copy()
    for c in key_cols:
        sub[c] = sub[c].astype(str).str.strip().replace({"nan": ""})
    df["_KEY_"] = sub.agg("||".join, axis=1)
    return df

for d in [dt_fn, rf_fn, log_fn, svm_fn]:
    build_key(d, KEY_COLS)

common_keys = set(dt_fn["_KEY_"]) & set(rf_fn["_KEY_"]) & set(log_fn["_KEY_"]) & set(svm_fn["_KEY_"])
print(f"共通で偽陰性となった件数: {len(common_keys)}")

base = dt_fn[dt_fn["_KEY_"].isin(common_keys)].copy()
base.drop(columns=["_KEY_"], inplace=True, errors="ignore")
base.to_csv("Common_FN_all_models.csv", index=False, encoding="utf-8-sig")

summary_cols = [
    "組合", "企業名", "期数",
    "所在地(実習先)(都道府県)", "所在地(実習先)(市区町村)",
    "派遣会社", "所属機関", "性別", "入国時年齢",
    "学校所属", "職種関係", "職種", "作業",
    "入国日", "失踪日", "失踪までの在日日数"
]

out_dir = Path("CommonFN_feature_counts")
out_dir.mkdir(exist_ok=True)

pd.set_option("display.max_rows", None)

for col in summary_cols:
    if col in base.columns:
        print(f"\n共通FNの{col}分布")
        vc = base[col].fillna("(欠損)").value_counts(dropna=False)
        print(vc)  # 显示所有行
        vc.to_csv(out_dir / f"CommonFN_counts_{col}.csv", header=["count"], encoding="utf-8-sig")
