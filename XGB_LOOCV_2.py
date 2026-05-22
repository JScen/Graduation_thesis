import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline
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

loo = LeaveOneOut()

# 不均衡比率（非失踪 / 失踪）→ scale_pos_weight に使う
n_neg = int((y == 0).sum())
n_pos = int((y == 1).sum())
spw = n_neg / n_pos
print(f"非失踪 {n_neg} 件 / 失踪 {n_pos} 件　→ scale_pos_weight = {spw:.2f}\n")


def evaluate(name, y_pred, y_prob):
    print(f"【{name}：LOOCV 結果】")
    print(classification_report(y, y_pred, digits=3))
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    print("混同行列:\n", cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    print("正規化混同行列:\n", np.round(cm_norm, 3))
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Sensitivity（感度）: {sens:.3f}")
    print(f"Specificity（特異度）: {spec:.3f}")
    print(f"AUC: {roc_auc_score(y, y_prob):.3f}\n")


# ====================================================================
# 版本 A：XGBoost + scale_pos_weight（他のモデルと同じ条件で公平に比較）
#   ※ class_weight に相当。SMOTE は使わない
# ====================================================================
clf_A = SkPipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,           # ← 不均衡対応（class_weight 相当）
        eval_metric="logloss", random_state=42, n_jobs=-1
    ))
])

print("=" * 60)
print("XGBoost + scale_pos_weight")
print("=" * 60)
yA = cross_val_predict(clf_A, X, y, cv=loo, n_jobs=-1)
yA_prob = cross_val_predict(clf_A, X, y, cv=loo,
                            method="predict_proba", n_jobs=-1)[:, 1]
evaluate("XGBoost + scale_pos_weight", yA, yA_prob)


