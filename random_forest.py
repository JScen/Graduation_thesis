import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# データ読み込み
df = pd.read_csv("clean.csv")

y = df["失踪の有無"]
X = df.drop(columns=["失踪の有無"])

sf = ["入国時年齢", "失踪までの在日日数"]
cf = [c for c in X.columns if c not in sf]

# 学習用とテスト用
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 前処理
preprocess = ColumnTransformer([
    ("num", "passthrough", sf),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cf)
])

# ランダムフォレスト
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

# 予測
y_pred = rf_clf.predict(X_test)
y_prob = rf_clf.predict_proba(X_test)[:, 1]

# 評価
print("\nランダムフォレスト")
print(classification_report(y_test, y_pred, digits=3))
print("混同行列:\n", confusion_matrix(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

# 特徴量重要度
feature_names = sf + list(
    rf_clf.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cf)
)
importances = rf_clf.named_steps["clf"].feature_importances_
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

print("\n特徴量重要度:")
for name, score in feat_imp[:10]:
    print(f"{name}: {score:.4f}")