import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# データ読み込み
df = pd.read_csv("clean.csv")
y = df["失踪の有無"].astype(int)
X = df.drop(columns=["失踪の有無"])

# 数値列とカテゴリ列
sf = [c for c in ["入国時年齢"] if c in X.columns]
cf = [c for c in X.columns if c not in sf]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=40, stratify=y
)

# 前処理
preprocess = ColumnTransformer([
    ("num", "passthrough", sf),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cf)
])

# 決定木モデル
dt_clf = Pipeline([
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        random_state=42,
        class_weight="balanced",
    ))
])

# 学習
dt_clf.fit(X_train, y_train)

y_pred_train = dt_clf.predict(X_train)
y_prob_train = dt_clf.predict_proba(X_train)[:, 1]

print("\n訓練データ結果")
print(classification_report(y_train, y_pred_train, digits=3))
print("混同行列（訓練データ）:\n", confusion_matrix(y_train, y_pred_train))
print("AUC（訓練データ）:", roc_auc_score(y_train, y_prob_train))

y_pred_test = dt_clf.predict(X_test)
y_prob_test = dt_clf.predict_proba(X_test)[:, 1]

print("\nテストデータ結果")
print(classification_report(y_test, y_pred_test, digits=3))
print("混同行列（テストデータ）:\n", confusion_matrix(y_test, y_pred_test))
print("AUC（テストデータ）:", roc_auc_score(y_test, y_prob_test))

# from sklearn.model_selection import LeaveOneOut, cross_val_predict
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
#
# loo = LeaveOneOut()
#
# y_pred_loo = cross_val_predict(dt_clf, X, y, cv=loo, n_jobs=-1)
# y_prob_loo = cross_val_predict(dt_clf, X, y, cv=loo, method="predict_proba", n_jobs=-1)[:, 1]
#
# print("\nLOOCV(tree)")
# print(classification_report(y, y_pred_loo, digits=3))
#
# cm = confusion_matrix(y, y_pred_loo, labels=[0,1])
# tn, fp, fn, tp = cm.ravel()
# sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
# specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#
# print("混同行列（LOOCV）:\n", cm)
# print(f"Sensitivity（感度）: {sensitivity:.3f}")
# print(f"Specificity（特異度）: {specificity:.3f}")
#
# print("ROC-AUC（LOOCV, pooled）:", roc_auc_score(y, y_prob_loo))

#特徴量重要度
print("\n=== 特徴量重要度（全て） ===")
ohe = dt_clf.named_steps["prep"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(cf)) if len(cf) else []
feature_names = sf + cat_names
importances = dt_clf.named_steps["clf"].feature_importances_
feat_imp_df = pd.DataFrame({
    "特徴量": feature_names,
    "重要度": importances
}).sort_values("重要度", ascending=False)

feat_imp_df.to_csv("tree.csv", index=False, encoding="utf-8-sig")
print("\n重要度を保存しました")