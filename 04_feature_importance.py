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

CSV_PATH = "clean2.csv"
TARGET_COL = "失踪の有無"
NUM_COL_CANDIDATES = ["入国時年齢"]
TOP_N = 30
RANDOM_STATE = 42

SVM_KERNEL = "linear"
SVM_C = 1.0

df = pd.read_csv(CSV_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"目的変数列 '{TARGET_COL}' が見つかりません: {df.columns}")

y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])

num_cols = [c for c in NUM_COL_CANDIDATES if c in X.columns]
cat_cols = [c for c in X.columns if c not in num_cols]

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocess_ohe = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", ohe, cat_cols),
    ],
    remainder="drop"
)

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

dt = DecisionTreeClassifier(
    max_depth=5,
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
    solver="liblinear",
    class_weight="balanced",
    max_iter=2000
)

svm = SVC(
    kernel=SVM_KERNEL,
    C=SVM_C,
    class_weight="balanced",
    probability=False
)

pipe_dt = Pipeline([("prep", preprocess_ohe), ("clf", dt)])
pipe_rf = Pipeline([("prep", preprocess_ohe), ("clf", rf)])

pipe_lr = Pipeline([("prep", preprocess_ohe), ("scaler", StandardScaler(with_mean=False)), ("clf", lr)])
pipe_svm = Pipeline([("prep", preprocess_ohe), ("scaler", StandardScaler(with_mean=False)), ("clf", svm)])

pipe_dt.fit(X, y)
feature_names = get_feature_names(pipe_dt.named_steps["prep"])

pipe_rf.fit(X, y)
pipe_lr.fit(X, y)
pipe_svm.fit(X, y)

def save_top_importances(model_name: str, scores: np.ndarray, names: list[str], top_n: int = TOP_N):
    df_out = pd.DataFrame({"feature": names, "score": scores})
    df_out["abs_score"] = df_out["score"].abs()
    df_out = df_out.sort_values("abs_score", ascending=False).head(top_n)
    out_path = f"{model_name}_top_features.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f" saved: {out_path}")
    return df_out

# 決定木重要度
dt_importances = pipe_dt.named_steps["clf"].feature_importances_
save_top_importances("DecisionTree", dt_importances, feature_names)

# ランダムフォレスト重要度
rf_importances = pipe_rf.named_steps["clf"].feature_importances_
save_top_importances("RandomForest", rf_importances, feature_names)

# ロジスティック回帰係数
lr_coef = pipe_lr.named_steps["clf"].coef_.ravel()  # shape (n_features,)
save_top_importances("LogisticRegression", lr_coef, feature_names)

# SVM係数
if SVM_KERNEL == "linear":
    svm_coef = pipe_svm.named_steps["clf"].coef_.ravel()
    save_top_importances("SVM_linear", svm_coef, feature_names)
else:
    X_trans = pipe_svm.named_steps["prep"].transform(X)
    X_trans = pipe_svm.named_steps["scaler"].transform(X_trans)
    svm_core = pipe_svm.named_steps["clf"]
    result = permutation_importance(
        svm_core,
        X_trans,
        y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="f1"
    )
    svm_perm = result.importances_mean
    save_top_importances("SVM_rbf_perm", svm_perm, feature_names)
