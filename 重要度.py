import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.inspection import permutation_importance


# =========================
# 1) 設定
# =========================
CSV_PATH = "clean2.csv"
TARGET_COL = "失踪の有無"
NUM_COL_CANDIDATES = ["入国時年齢"]  # 数値列候補（なければ無視）
TOP_N = 30                           # 上位いくつ保存するか
RANDOM_STATE = 42

# SVMのカーネル設定
SVM_KERNEL = "linear"  # "linear" or "rbf"
SVM_C = 1.0


# =========================
# 2) データ読込
# =========================
df = pd.read_csv(CSV_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"目的変数列 '{TARGET_COL}' が見つかりません: {df.columns}")

y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])

# 数値列とカテゴリ列を自動判定
num_cols = [c for c in NUM_COL_CANDIDATES if c in X.columns]
cat_cols = [c for c in X.columns if c not in num_cols]


# =========================
# 3) 前処理（One-Hotを統一）
#    - LR/SVMは標準化も入れる（数値＋OneHot後）
# =========================
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocess_ohe = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", ohe, cat_cols),
    ],
    remainder="drop"
)

# 変換後の特徴量名を取り出すユーティリティ
def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = []
    # num
    if num_cols:
        names.extend(num_cols)
    # cat
    if cat_cols:
        ohe_fitted = preprocessor.named_transformers_["cat"]
        names.extend(list(ohe_fitted.get_feature_names_out(cat_cols)))
    return names


# =========================
# 4) モデル定義
# =========================
dt = DecisionTreeClassifier(
    max_depth=5,                 # 可視化/解釈重視（必要なら変える）
    class_weight="balanced",
    random_state=RANDOM_STATE
)

rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

lr = LogisticRegression(
    solver="liblinear",          # 二値で安定
    class_weight="balanced",
    max_iter=2000
)

svm = SVC(
    kernel=SVM_KERNEL,
    C=SVM_C,
    class_weight="balanced",
    probability=False
)

# 木系（標準化なし）
pipe_dt = Pipeline([("prep", preprocess_ohe), ("clf", dt)])
pipe_rf = Pipeline([("prep", preprocess_ohe), ("clf", rf)])

# 線形系（標準化あり）
# OneHot後に標準化：with_mean=False は疎行列対策だが、ここは dense 出力なので通常OK。
# 念のため with_mean=False にしておくと安全。
pipe_lr = Pipeline([("prep", preprocess_ohe), ("scaler", StandardScaler(with_mean=False)), ("clf", lr)])
pipe_svm = Pipeline([("prep", preprocess_ohe), ("scaler", StandardScaler(with_mean=False)), ("clf", svm)])


# =========================
# 5) 学習して「特徴量名」を確定
# =========================
# 特徴量名を確定させるために一度fit（全データでOK：抽出目的）
pipe_dt.fit(X, y)
feature_names = get_feature_names(pipe_dt.named_steps["prep"])

# 他モデルもfit（係数/重要度抽出のため）
pipe_rf.fit(X, y)
pipe_lr.fit(X, y)
pipe_svm.fit(X, y)


# =========================
# 6) 抽出関数
# =========================
def save_top_importances(model_name: str, scores: np.ndarray, names: list[str], top_n: int = TOP_N):
    """scores: 大きいほど重要（符号は保持してもOK）"""
    df_out = pd.DataFrame({"feature": names, "score": scores})
    df_out["abs_score"] = df_out["score"].abs()
    df_out = df_out.sort_values("abs_score", ascending=False).head(top_n)
    out_path = f"{model_name}_top_features.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ saved: {out_path}")
    return df_out


# =========================
# 7) 決定木：重要度
# =========================
dt_importances = pipe_dt.named_steps["clf"].feature_importances_
save_top_importances("DecisionTree", dt_importances, feature_names)

# =========================
# 8) ランダムフォレスト：重要度
# =========================
rf_importances = pipe_rf.named_steps["clf"].feature_importances_
save_top_importances("RandomForest", rf_importances, feature_names)

# =========================
# 9) ロジスティック回帰：係数（正=失踪リスク↑）
# =========================
lr_coef = pipe_lr.named_steps["clf"].coef_.ravel()  # shape (n_features,)
save_top_importances("LogisticRegression", lr_coef, feature_names)

# =========================
# 10) SVM：線形なら係数 / 非線形なら permutation importance
# =========================
if SVM_KERNEL == "linear":
    svm_coef = pipe_svm.named_steps["clf"].coef_.ravel()
    save_top_importances("SVM_linear", svm_coef, feature_names)
else:
    # permutation importance は計算が重いので、まず全データfit後に実行
    # scoring は recall を重視するなら "recall" などに変更可能
    # 例: scoring="recall"（失踪=1を重視）
    X_trans = pipe_svm.named_steps["prep"].transform(X)
    X_trans = pipe_svm.named_steps["scaler"].transform(X_trans)

    # SVC本体に対して permutation をかける（前処理後）
    svm_core = pipe_svm.named_steps["clf"]
    result = permutation_importance(
        svm_core,
        X_trans,
        y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="f1"  # 必要なら "recall" に
    )
    svm_perm = result.importances_mean
    save_top_importances("SVM_rbf_perm", svm_perm, feature_names)

print("\n=== DONE ===")