import pandas as pd

# Excelファイル読み込み
file = "excel.xlsx"
df = pd.read_excel(file)

# 1. 都道府県のユニーク数
num_pref = df["所在地(実習先)(都道府県)"].nunique()

# 2. 市区町村のユニーク数 & 町村の割合
unique_city = df["所在地(実習先)(市区町村)"].dropna().unique()
num_city = len(unique_city)
num_town_village = sum(("町" in str(x) or "村" in str(x)) for x in unique_city)
ratio_town_village = num_town_village / num_city * 100

# 3. 失踪者の都道府県別人数と「都道府県ごとの全実習生に占める割合」
total_by_pref = df.groupby("所在地(実習先)(都道府県)").size()  # 各都道府県の総人数
lost_by_pref = df[df["状況のまとめ"] == "失踪"].groupby("所在地(実習先)(都道府県)").size()  # 失踪人数

lost_summary = pd.DataFrame({
    "全体人数": total_by_pref,
    "失踪者数": lost_by_pref
}).fillna(0)  # NaNは0にする
lost_summary["失踪率(%)"] = (lost_summary["失踪者数"] / lost_summary["全体人数"] * 100).round(2)

# 結果出力
print("都道府県の数:", num_pref)
print("市区町村の数:", num_city)
print("町村の数:", num_town_village, f"({ratio_town_village:.1f}%)")
print("\n失踪者の都道府県別人数とその県の全実習生に対する割合:")
print(lost_summary)