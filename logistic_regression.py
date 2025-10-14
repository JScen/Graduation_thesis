import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# データ読み込み
df = pd.read_csv("clean.csv")

y = df["失踪の有無"].astype(int)
X = df.drop(columns=["失踪の有無"])

sf = [c for c in ["入国時年齢"] if c in X.columns]
cf = [c for c in X.columns if c not in sf]

# 学習用とテスト用
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 前処理
preprocess = ColumnTransformer([
    ("num", StandardScaler(), sf),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cf)
])

# ロジスティック回帰モデル
log_clf = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

# 学習
log_clf.fit(X_train, y_train)

# 予測
y_pred = log_clf.predict(X_test)
y_prob = log_clf.predict_proba(X_test)[:, 1]

# 評価
print("\n=== ロジスティック回帰 ===")
print(classification_report(y_test, y_pred, digits=3))
print("混同行列:\n", confusion_matrix(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

# 特徴量の係数確認
ohe = log_clf.named_steps["prep"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(cf)) if len(cf) else []
feature_names = sf + cat_names
coef = log_clf.named_steps["clf"].coef_[0]

coef_df = pd.DataFrame({
    "特徴量": feature_names,
    "係数": coef
}).sort_values("係数", ascending=False)

print("\n=== 係数高===")
print(coef_df.head(10).to_string(index=False))
print("\n=== 係数低===")
print(coef_df.tail(10).to_string(index=False))