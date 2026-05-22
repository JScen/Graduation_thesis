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
        kernel="linear",
        C=1.0,
        class_weight="balanced",
        probability=True,
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

y_pred_train = svm_clf.predict(X_train)
y_prob_train = svm_clf.predict_proba(X_train)[:, 1]

print("\nSVM_train")
print(classification_report(y_train, y_pred_train, digits=3))
print("混同行列（訓練データ）:\n", confusion_matrix(y_train, y_pred_train))
print("AUC（訓練データ）:", roc_auc_score(y_train, y_prob_train))

print("\nSVM_test")
print(classification_report(y_test, y_pred, digits=3))
print("混同行列:\n", confusion_matrix(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

pred_df = X_test.copy()
pred_df["予測結果"] = y_pred
pred_df["失踪数値"] = y_prob

pred_df.to_csv("svmkakuritu_test.csv", index=False, encoding="utf-8-sig")
print("\nsvmkakuritu_testを保存しました")

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

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd

loo = LeaveOneOut()

y_pred_loo = cross_val_predict(svm_clf, X, y, cv=loo, n_jobs=-1)

y_prob_loo = cross_val_predict(svm_clf, X, y, cv=loo, method="predict_proba", n_jobs=-1)[:, 1]

print("\nLOOCV")
print(classification_report(y, y_pred_loo, digits=3))

cm = confusion_matrix(y, y_pred_loo, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print("混同行列（LOOCV）:\n", cm)

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
print(f"Sensitivity（感度）: {sensitivity:.3f}")
print(f"Specificity（特異度）: {specificity:.3f}")

auc_loo = roc_auc_score(y, y_prob_loo)
print("ROC-AUC（LOOCV, pooled）:", auc_loo)

result_df = X.copy()
result_df["実際(失踪の有無)"] = y
result_df["予測(LOOCV)"] = y_pred_loo
result_df["失踪確率(LOOCV)"] = y_prob_loo

fn_df = result_df[(result_df["実際(失踪の有無)"] == 1) & (result_df["予測(LOOCV)"] == 0)]

fn_df.to_csv("LOOCV_FN_Svm.csv", index=False, encoding="utf-8-sig")
print("偽陰性一覧を 'LOOCV_FN_Svm.csv' に保存しました。")