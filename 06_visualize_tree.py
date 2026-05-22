import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("clean2.csv")
y = df["失踪の有無"].astype(int)
X = df.drop(columns=["失踪の有無"])

num_cols = ["入国時年齢"]
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = Pipeline([
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(
        max_depth=4,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X, y)

ohe = model.named_steps["prep"].named_transformers_["cat"]
feature_names = num_cols + list(ohe.get_feature_names_out(cat_cols))

plt.figure(figsize=(28, 14))
plot_tree(
    model.named_steps["clf"],
    feature_names=feature_names,
    class_names=["非失踪", "失踪"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
plt.close()