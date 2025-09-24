from main import pd, plt, load_data

df = load_data()
df_filtered = df[df["状況のまとめ"] == "失踪"]
df_grouped = (
    df_filtered
    .groupby("職種関係")["名前"]
    .count()
    .reset_index()
    .rename(columns={"名前": "失踪者数"})
    .dropna()
)

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({val}人)"
    return my_autopct

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    df_grouped["失踪者数"],
    autopct=make_autopct(df_grouped["失踪者数"]),
    startangle=140,
    counterclock=False,
    pctdistance=1.1,
    textprops={'fontsize': 10}
)
plt.legend(
    wedges,
    df_grouped["職種関係"],
    title="職種関係",
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)
plt.title("職種関係別の失踪者の割合", fontsize=14)
plt.tight_layout()
plt.show()
