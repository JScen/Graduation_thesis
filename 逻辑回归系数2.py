import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ==============================
# データ読み込み
# ==============================
df = pd.read_csv("clean2.csv")

y = df["失踪の有無"].astype(int)
X = df.drop(columns=["失踪の有無"])

# ==============================
# 数値変数・カテゴリ変数
# ==============================
num_cols = ["入国時年齢"]

cat_cols = [
    "性別",
    "職種関係",
    "職種",
    "所在地(実習先)(都道府県)",
    "所在地(実習先)(市区町村)",
    "派遣会社",
    "学校所属",
    "組合",
    "所属機関"
]

# ==============================
# 前処理
# ==============================
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# ==============================
# Logistic 回帰モデル
# ==============================
logreg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear"
)

pipe = Pipeline([
    ("prep", preprocess),
    ("clf", logreg)
])

# ==============================
# 学習（全データで）
# ==============================
pipe.fit(X, y)

# ==============================
# 特徴量名の取得
# ==============================
ohe = pipe.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)

feature_names = num_cols + list(cat_feature_names)

# ==============================
# 係数の取得
# ==============================
coef = pipe.named_steps["clf"].coef_[0]

coef_df = pd.DataFrame({
    "特徴量": feature_names,
    "係数": coef,
    "絶対値": np.abs(coef)
})

# 絶対値順で並び替え
coef_df = coef_df.sort_values("絶対値", ascending=False)

# ==============================
# CSV 出力
# ==============================
coef_df.to_csv(
    "LogisticRegression_系数.csv",
    index=False,
    encoding="utf-8-sig"
)

print("✅ Logistic 回帰の係数を出力しました")
print("➡ LogisticRegression_coefficients.csv")