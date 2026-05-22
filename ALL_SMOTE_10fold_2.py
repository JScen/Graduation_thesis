import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline   # ← imblearn の Pipeline
from xgboost import XGBClassifier

# ============ データ読み込み ============
df = pd.read_csv("clean2.csv")
y = df["失踪の有無"].astype(int)
X = df.drop(columns=["失踪の有無"])

sf = ["入国時年齢"]
cf = [c for c in X.columns if c not in sf]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), sf),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cf)
])

# ============ 統一した評価方法：10-fold 層化交差検証 ============
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ============ 5モデル（全て SMOTE + 10-fold で統一） ============
# 注意：SMOTE を使うので class_weight / scale_pos_weight は設定しない
#       （二重に不均衡対応すると過補正になるため）
models = {
    "DecisionTree": DecisionTreeClassifier(
        criterion="gini", random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=2000, solver="lbfgs"
    ),
    "SVM": SVC(
        kernel="linear", probability=True, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1
    ),
}

# 結果をまとめる辞書
summary = []


def run_model(name, model):
    print("=" * 60)
    print(f"【{name} + SMOTE：10-fold Stratified CV 結果】")
    print("=" * 60)

    clf = Pipeline([
        ("prep", preprocess),
        ("smote", SMOTE(random_state=42)),
        ("clf", model)
    ])

    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    y_prob = cross_val_predict(clf, X, y, cv=cv,
                               method="predict_proba", n_jobs=-1)[:, 1]

    print(classification_report(y, y_pred, digits=3))

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    print("混同行列:\n", cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    print("正規化混同行列:\n", np.round(cm_norm, 3))

    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    auc = roc_auc_score(y, y_prob)

    print(f"Sensitivity（感度）: {sens:.3f}")
    print(f"Specificity（特異度）: {spec:.3f}")
    print(f"AUC: {auc:.3f}\n")

    summary.append({
        "Model": name,
        "Sensitivity": round(sens, 3),
        "Specificity": round(spec, 3),
        "F1(失踪)": round(f1_pos, 3),
        "AUC": round(auc, 3),
    })


for name, model in models.items():
    run_model(name, model)

# ============ 最終比較表 ============
print("=" * 60)
print("【全モデル比較（SMOTE + 10-fold Stratified CV）】")
print("=" * 60)
result_df = pd.DataFrame(summary).sort_values("Sensitivity", ascending=False)
print(result_df.to_string(index=False))
result_df.to_csv("model_comparison_SMOTE_10fold.csv",
                 index=False, encoding="utf-8-sig")
print("\n比較表を model_comparison_SMOTE_10fold.csv に保存しました")
