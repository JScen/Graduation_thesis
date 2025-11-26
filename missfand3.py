import pandas as pd

df = pd.read_csv("clean.csv")

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

# ★ 各グループの人数（0 + 1）を追加
group_stats["人数"] = group_stats[0] + group_stats[1]

# 失踪と未失踪が両方いる重複グループ（人数は自然に2人以上になる）
mixed_groups = group_stats[(group_stats[0] > 0) & (group_stats[1] > 0)]

mixed_total = mixed_groups["人数"].sum()

print("失踪と未失踪重複グループ")
print(f"件数: {len(mixed_groups)}合計人数: {mixed_total}")

# 全員失踪かつ 2人以上いる重複グループ
all_lost_groups = group_stats[
    (group_stats[0] == 0) & (group_stats[1] > 0) & (group_stats["人数"] >= 2)
]

mixed_total2 = all_lost_groups["人数"].sum()

print("全員失踪重複グループ")
print(f"件数: {len(all_lost_groups)}合計人数: {mixed_total2}")

# 全員未失踪かつ 2人以上いる重複グループ
all_safe_groups = group_stats[
    (group_stats[0] > 0) & (group_stats[1] == 0) & (group_stats["人数"] >= 2)
]

mixed_total3 = all_safe_groups["人数"].sum()

print("全員未失踪重複グループ")
print(f"件数: {len(all_safe_groups)}合計人数: {mixed_total3}")

mixed_groups.to_csv("jyuhukumix.csv", encoding="utf-8-sig")
all_lost_groups.to_csv("jyuhukulost.csv", encoding="utf-8-sig")
all_safe_groups.to_csv("jyuhukusafe.csv", encoding="utf-8-sig")

print("\nCSV 出力完了：")

conflict_detail = df.merge(
    mixed_groups.reset_index()[feature_cols],
    on=feature_cols,
    how="inner"
)

conflict_detail.to_csv("jyuhuku.csv", index=False, encoding="utf-8-sig")

print("\njyuhuku.csv out")