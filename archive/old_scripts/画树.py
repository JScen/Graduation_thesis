import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# データ読み込み
df = pd.read_csv("clean2.csv")

y = df["失踪の有無"].astype(int)
X = df.drop(columns=["失踪の有無"])

# 数値列とカテゴリ列
sf = [c for c in ["入国時年齢"] if c in X.columns]
cf = [c for c in X.columns if c not in sf]

# 前処理（カテゴリはOne-Hot）
preprocess = ColumnTransformer([
    ("num", "passthrough", sf),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cf)
])

# 決定木（読みやすさのため深さ制限）
dt_clf = Pipeline([
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    ))
])

# 学習（全データでOK：木の構造を見る目的）
dt_clf.fit(X, y)

# --- ここがポイント：One-Hot後の特徴量名を取得 ---
ohe = dt_clf.named_steps["prep"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(cf)) if len(cf) else []
feature_names = sf + cat_names

# --- 決定木本体を取り出して文字化 ---
tree_model = dt_clf.named_steps["clf"]

tree_text = export_text(
    tree_model,
    feature_names=feature_names
)

print(tree_text)

# 保存
with open("decision_tree_text.txt", "w", encoding="utf-8") as f:
    f.write(tree_text)

print("\n✅ decision_tree_text.txt に保存しました")