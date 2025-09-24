from main import pd, plt, load_data

df = load_data()
days = df["失踪までの在日日数"].dropna().astype(int)

bins = 10
counts, bin_edges, patches = plt.hist(
    days,
    bins=bins,
    edgecolor='black',
    zorder=1
)

mean_val = days.mean()
median_val = days.median()
max_count = counts.max()

for count, left, right in zip(counts, bin_edges[:-1], bin_edges[1:]):
    x = (left + right) / 2
    y = count + max_count * 0.01
    plt.text(x, y, f"{int(left)}–{int(right)}", ha='center', fontsize=9)

plt.axvline(mean_val, color='red', linestyle='--', label=f"平均: {int(mean_val)}日", zorder=3)
plt.axvline(median_val, color='green', linestyle=':', label=f"中央値: {int(median_val)}日", zorder=3)

plt.axvline(365, color='orange', linestyle='-', linewidth=1.5, label='365日', zorder=2)
plt.axvline(730, color='black', linestyle='-', linewidth=1.5, label='730日', zorder=2)

plt.legend()
plt.xlabel("在日日数（日）")
plt.ylabel("失踪者数")
plt.tight_layout()
plt.show()
