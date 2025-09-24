from main import pd, plt, load_data
import seaborn as sns

df = load_data()
subset = df[["入国時年齢", "失踪までの在日日数"]].dropna()
subset["入国時年齢"] = subset["入国時年齢"].astype(int)
subset["失踪までの在日日数"] = subset["失踪までの在日日数"].astype(int)

plt.figure(figsize=(10, 6))
sns.regplot(
    data=subset,
    x="入国時年齢",
    y="失踪までの在日日数",
    scatter_kws={"alpha": 0.6},
    line_kws={"color": "red"}
)
plt.title("入国時年齢と失踪までの在日日数の関係", fontsize=14)
plt.xlabel("入国時年齢（歳）")
plt.ylabel("失踪までの在日日数（日）")
plt.grid(True)
plt.tight_layout()
plt.show()
