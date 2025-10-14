import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_excel("deta.xlsx")

#データ数
print("データ数:", df.shape)

#バックアップ
df.to_csv("backup_raw.csv", index=False, encoding="utf-8-sig")

#データ抽出(状況のまとめ)
print(df["状況のまとめ"].astype(str).value_counts().head(10))

#(目標変数カテゴリ作り)
df["失踪の有無"] = df["状況のまとめ"].astype(str).str.contains("失踪", na=False).astype("int8")

#入国待ちとその他を排除
df_f = df[~df["状況のまとめ"].isin(["入国待ち", "その他"])].copy()
print("入国待ちとその他を排除後データ数:", df_f.shape)
print("排除後 状況のまとめ:")
print(df_f["状況のまとめ"].value_counts())

#失踪率
print("失踪率：", df_f["失踪の有無"].mean())

#テスト
#print(df_f[["状況のまとめ","失踪の有無"]].head(10))

#目標変数
y = df_f["失踪の有無"]

#説明変数
#suuji_features = ["入国時年齢", "失踪までの在日日数"]
suuji_features = ["入国時年齢"]
categori_features = ["性別", "職種関係", "職種", "所在地(実習先)(都道府県)", "学校所属" ,"組合"]
sf = suuji_features
cf = categori_features

X = df_f[sf + cf].copy()

# 欠損値補完
X["入国時年齢"] = X["入国時年齢"].fillna(X["入国時年齢"].median())
#X["失踪までの在日日数"] = X["失踪までの在日日数"].fillna(0)
for col in cf:
    X[col] = X[col].astype("string").fillna("N/A")

# 学習用とテスト用
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# test
print("学習データサイズ:", X_train.shape)
print("テストデータサイズ:", X_test.shape)
print("学習データ失踪率:", y_train.mean())
print("テストデータ失踪率:", y_test.mean())

# 学習用データ保存
clean = X.copy()
clean["失踪の有無"] = y
clean.to_csv("clean.csv", index=False, encoding="utf-8-sig")