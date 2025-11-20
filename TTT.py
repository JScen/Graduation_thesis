import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = [
    'Hiragino Sans',
    'IPAexGothic',
    'Noto Sans CJK JP',
    'Meiryo',
    'Arial Unicode MS'
]
rcParams['axes.unicode_minus'] = False

df = pd.read_csv("clean.csv")
y = df["失踪の有無"].astype(int)
X = df.drop(columns=["失踪の有無"])

sf = [c for c in ["入国時年齢"] if c in X.columns]
cf = [c for c in X.columns if c not in sf]

preprocess = ColumnTransformer([
    ("num", "passthrough", sf),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cf)
])

dt_clf = Pipeline([
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        class_weight="balanced",
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=40, stratify=y
)
dt_clf.fit(X_train, y_train)

ohe = dt_clf.named_steps["prep"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(cf)) if len(cf) else []
feature_names = sf + cat_names

tree = dt_clf.named_steps["clf"]

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.figure(figsize=(48, 48), dpi=200)
plot_tree(
    tree,
    feature_names=feature_names,
    class_names=["未失踪(0)", "失踪(1)"],
    filled=True,
    rounded=True,
    impurity=True,
    proportion=False,
    precision=2
)
plt.tight_layout()
plt.savefig("DecisionTree_full.pdf", bbox_inches="tight")
plt.savefig("DecisionTree_full.png", bbox_inches="tight")
plt.close()

print("決定木の全体図を保存しました")