import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("clean.csv")
y = df["失踪の有無"].astype(int)
X = df.drop(columns=["失踪の有無"])

sf = [c for c in ["入国時年齢"] if c in X.columns]
cf = [c for c in X.columns if c not in sf]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), sf),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cf)
])

svm_clf = Pipeline([
    ("prep", preprocess),
    ("clf", SVC(
        kernel="linear",          # 線形SVM（ロジスティック回帰に近い）
        C=1.0,                    # 正則化パラメータ
        class_weight="balanced",  # 不均衡データ対応
        probability=True,         # 確率出力を有効化（AUC計算に必要）
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=40, stratify=y
)

# 学習
svm_clf.fit(X_train, y_train)

# 予測
y_pred = svm_clf.predict(X_test)
y_prob = svm_clf.predict_proba(X_test)[:, 1]

# 評価
print("\nSVM")
print(classification_report(y_test, y_pred, digits=3))
print("混同行列:\n", confusion_matrix(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

clf_inner = svm_clf.named_steps["clf"]
ohe = svm_clf.named_steps["prep"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(cf)) if len(cf) else []
feature_names = sf + cat_names

import numpy as np
coef = np.ravel(clf_inner.coef_.toarray())

coef_df = pd.DataFrame({
    "特徴量": feature_names,
    "係数": coef
}).sort_values("係数", ascending=False)

coef_df.to_csv("svm.csv", index=False, encoding="utf-8-sig")
print("\n結果を保存しました")