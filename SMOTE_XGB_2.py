import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline   # ← imblearn の Pipeline（sklearn のではない）
from xgboost import XGBClassifier

# ============ データ読み込み ============
df = pd.read_csv("clean2.csv")
y = df["失踪の有無"].astype(int)
X = df.drop(columns=["失踪の有無"])

# 数値列・カテゴリ列
sf = ["入国時年齢"]
cf = [c for c in X.columns if c not in sf]

# 前処理（数値は標準化，カテゴリは One-Hot）
preprocess = ColumnTransformer([
    ("num", StandardScaler(), sf),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cf)
])

# ============ 10-fold 層化交差検証 ============
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ============ SMOTE + XGBoost パイプライン ============
# 重要：SMOTE は Pipeline 内に入れることで，各 fold の「訓練データのみ」に適用される
#       （データリークを防ぐ）
clf = Pipeline([
    ("prep", preprocess),
    ("smote", SMOTE(random_state=42)),
    ("clf", XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
        # 注意：SMOTE を使うので scale_pos_weight は設定しない
    ))
])

print("【XGBoost + SMOTE：10-fold Stratified CV 結果】")

# 全サンプルの予測を集約（cross_val_predict）
y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
y_prob = cross_val_predict(clf, X, y, cv=cv,
                           method="predict_proba", n_jobs=-1)[:, 1]

# ============ 評価 ============
print(classification_report(y, y_pred, digits=3))

cm = confusion_matrix(y, y_pred, labels=[0, 1])
print("混同行列:\n", cm)

# 正規化混同行列（行ごとに割合化）
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
print("正規化混同行列（行=実際のクラス）:\n", np.round(cm_norm, 3))

tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"Sensitivity（感度）: {sensitivity:.3f}")
print(f"Specificity（特異度）: {specificity:.3f}")
print(f"AUC: {roc_auc_score(y, y_prob):.3f}")

# 偽陰性（実際は失踪だが非失踪と予測）を保存
fn_detail = df[(y == 1) & (y_pred == 0)]
fn_detail.to_csv("SMOTE_XGB_FN_2.csv", index=False, encoding="utf-8-sig")
print(f"\n偽陰性 {len(fn_detail)} 件を SMOTE_XGB_FN_2.csv に保存しました")
