import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("clean.csv")

y = df["失踪の有無"]
X = df.drop(columns=["失踪の有無"])

X = X[["入国時年齢", "失踪までの在日日数"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("\nランダムフォレスト")
print(classification_report(y_test, y_pred, digits=3))
print("AUC:", roc_auc_score(y_test, y_prob))