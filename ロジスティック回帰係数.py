import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("clean2.csv")

y = df["失踪の有無"].astype(int)
X = df.drop(columns=["失踪の有無"])

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

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

logreg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear"
)

pipe = Pipeline([
    ("prep", preprocess),
    ("clf", logreg)
])

pipe.fit(X, y)

ohe = pipe.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)

feature_names = num_cols + list(cat_feature_names)

coef = pipe.named_steps["clf"].coef_[0]

coef_df = pd.DataFrame({
    "特徴量": feature_names,
    "係数": coef,
    "絶対値": np.abs(coef)
})

coef_df = coef_df.sort_values("絶対値", ascending=False)

coef_df.to_csv(
    "LogisticRegression_系数.csv",
    index=False,
    encoding="utf-8-sig"
)