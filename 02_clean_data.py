import pandas as pd

df = pd.read_csv("clean.csv")

# 使用する特徴量
feature_cols = [
    "入国時年齢", "性別", "職種関係", "職種",
    "所在地(実習先)(都道府県)", "所在地(実習先)(市区町村)",
    "派遣会社", "学校所属", "組合", "所属機関"
]

group_stats = (
    df.groupby(feature_cols)["失踪の有無"]
      .value_counts()
      .unstack(fill_value=0)
)

# 失踪と未失踪が混在する矛盾グループ
mixed_groups = group_stats[(group_stats[0] > 0) & (group_stats[1] > 0)]
mixed_groups.to_csv("jyuhukumix.csv", encoding="utf-8-sig")

# 全員失踪
all_lost_groups = group_stats[(group_stats[0] == 0) & (group_stats[1] >= 2)]
all_lost_groups.to_csv("jyuhukulost.csv", encoding="utf-8-sig")

#  全員未失踪
all_safe_groups = group_stats[(group_stats[1] == 0) & (group_stats[0] >= 2)]
all_safe_groups.to_csv("jyuhukusafe.csv", encoding="utf-8-sig")

# 矛盾グループの詳細データ
conflict_detail = df.merge(
    mixed_groups.reset_index()[feature_cols],
    on=feature_cols,
    how="inner"
)

conflict_detail.to_csv("jyuhuku.csv", index=False, encoding="utf-8-sig")
print("\njyuhuku.csv out")

# 人数を表示

print("削除前の人数（clean.csv）")


total_before = len(df)
lost_before = df["失踪の有無"].sum()
safe_before = total_before - lost_before

print(f"総人数: {total_before}")
print(f"失踪人数: {lost_before}")
print(f"未失踪人数: {safe_before}")

df_clean2 = df.merge(
    mixed_groups.reset_index()[feature_cols],
    on=feature_cols,
    how="left",
    indicator=True
)

df_clean2 = df_clean2[df_clean2["_merge"] == "left_only"].drop(columns=["_merge"])

df_clean2.to_csv("clean2.csv", index=False, encoding="utf-8-sig")

print("矛盾グループ削除後の人数clean2.csv")


total_after = len(df_clean2)
lost_after = df_clean2["失踪の有無"].sum()
safe_after = total_after - lost_after

print(f"総人数: {total_after}")
print(f"失踪人数: {lost_after}")
print(f"未失踪人数: {safe_after}")
