import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# データ分割（層化抽出）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=40, stratify=y
)

# 前処理
preprocess = ColumnTransformer([
    ("num", "passthrough", sf),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cf)
])

# ランダムフォレストモデル
rf_clf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# 学習
rf_clf.fit(X_train, y_train)

y_pred_train = rf_clf.predict(X_train)
y_prob_train = rf_clf.predict_proba(X_train)[:, 1]

print("\n訓練データ果")
print(classification_report(y_train, y_pred_train, digits=3))
print("混同行列（訓練データ）:\n", confusion_matrix(y_train, y_pred_train))
print("AUC（訓練データ）:", roc_auc_score(y_train, y_prob_train))

y_pred_test = rf_clf.predict(X_test)
y_prob_test = rf_clf.predict_proba(X_test)[:, 1]

print("\nテストデータ結果")
print(classification_report(y_test, y_pred_test, digits=3))
print("混同行列（テストデータ）:\n", confusion_matrix(y_test, y_pred_test))
print("AUC（テストデータ）:", roc_auc_score(y_test, y_prob_test))

# 特徴量重要度
#feature_names = sf + list(
#    rf_clf.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cf)
#)
#importances = rf_clf.named_steps["clf"].feature_importances_
#feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
#
#print("\n=== 全特徴量の重要度 ===")
#for name, score in feat_imp:
#    print(f"{name}: {score:.4f}")