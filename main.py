import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_excel("deta.xlsx")
df = df.dropna(axis=1, how="all")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

#データ数
print("データ数:", df.shape)

#バックアップ
df.to_csv("backup_raw.csv", index=False, encoding="utf-8-sig")

#データ抽出(状況のまとめ)
print(df["状況のまとめ"].value_counts())

#(目標変数カテゴリ作り)
df["失踪の有無"] = df["状況のまとめ"].astype(str).str.contains("失踪", na=False).astype("int8")

#入国待ちとその他を排除
df_f = df[~df["状況のまとめ"].isin(["入国待ち", "その他"])].copy()
print("入国待ちとその他を排除後データ数:", df_f.shape)
print("排除後 状況のまとめ:")
print(df_f["状況のまとめ"].value_counts())

#失踪率
print("失踪率：", df_f["失踪の有無"].mean())

#目標変数
y = df_f["失踪の有無"]

#説明変数
#suuji_features = ["入国時年齢", "失踪までの在日日数"]
suuji_features = ["入国時年齢"]
categori_features = ["性別", "職種関係", "職種", "所在地(実習先)(都道府県)", "所在地(実習先)(市区町村)",
                     "派遣会社", "学校所属", "組合", "所属機関"]
sf = suuji_features
cf = categori_features

X = df_f[sf + cf].copy()

# 欠損値補完
X["入国時年齢"] = X["入国時年齢"].fillna(X["入国時年齢"].median())
for col in cf:
    X[col] = X[col].astype("string").fillna("N/A")

# 学習用データ保存
clean = X.copy()
clean["失踪の有無"] = y
clean.to_csv("clean.csv", index=False, encoding="utf-8-sig")