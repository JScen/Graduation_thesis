import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("clean2.csv")
y = df["失踪の有無"].astype(int)
X = df.drop(columns=["失踪の有無"])

# 数値列・カテゴリ列
sf = ["入国時年齢"]
cf = [c for c in X.columns if c not in sf]

# 前処理
preprocess = ColumnTransformer([
    ("num", StandardScaler(), sf),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cf)
])

loo = LeaveOneOut()

models = {
    "DecisionTree": DecisionTreeClassifier(
        criterion="gini", class_weight="balanced", random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs"
    ),
    "SVM": SVC(
        kernel="linear", probability=True, class_weight="balanced", random_state=42
    )
}

def run_loo(model_name, model):
    print(f"【{model_name}：LOOCV 結果】")
    clf = Pipeline([
        ("prep", preprocess),
        ("clf", model)
    ])

    y_pred = cross_val_predict(clf, X, y, cv=loo, n_jobs=-1)
    y_prob = cross_val_predict(clf, X, y, cv=loo,
                               method="predict_proba", n_jobs=-1)[:, 1]

    print(classification_report(y, y_pred, digits=3))

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    print("混同行列:\n", cm)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Sensitivity（感度）: {sensitivity:.3f}")
    print(f"Specificity（特異度）: {specificity:.3f}")
    print(f"AUC: {roc_auc_score(y, y_prob):.3f}")

    fn_detail = df[(y == 1) & (y_pred == 0)]
    fn_detail.to_csv(f"LOOCV_FN_{model_name}_2.csv",
                     index=False, encoding="utf-8-sig")

for name, model in models.items():
    run_loo(name, model)