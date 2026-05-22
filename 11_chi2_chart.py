"""
カイ二乗検定まとめ図（シンプル版）
テーブル形式で結果を一目で把握できるようにする
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import chi2_contingency, mannwhitneyu
import matplotlib.patches as mpatches

# =====================================
# 0. フォント設定（Mac用）
# =====================================
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

COLOR_FN     = "#e74c3c"
COLOR_DETECT = "#3498db"
COLOR_SAFE   = "#2ecc71"

# =====================================
# 1. データ読み込み
# =====================================
fn_df  = pd.read_csv("Common_FN_all_models_2.csv")
all_df = pd.read_csv("clean2.csv")

lost_all = all_df[all_df["失踪の有無"] == 1].copy()
non_lost = all_df[all_df["失踪の有無"] == 0].copy()

KEY_COLS = [
    "入国時年齢", "性別", "職種関係", "職種",
    "所在地(実習先)(都道府県)", "所在地(実習先)(市区町村)",
    "派遣会社", "学校所属", "組合", "所属機関"
]

def make_key(df, cols):
    sub = df[[c for c in cols if c in df.columns]].copy()
    for c in sub.columns:
        sub[c] = sub[c].astype(str).str.strip().replace({"nan": ""})
    return sub.agg("||".join, axis=1)

fn_keys     = set(make_key(fn_df, KEY_COLS))
lost_keys   = make_key(lost_all, KEY_COLS)
non_fn_lost = lost_all[~lost_keys.isin(fn_keys)].copy()

# =====================================
# 2. 検定結果の計算
# =====================================
cat_cols = ["性別", "職種関係", "職種", "所在地(実習先)(都道府県)",
            "派遣会社", "学校所属", "組合", "所属機関"]
cat_cols = [c for c in cat_cols if c in fn_df.columns]

rows = []
for col in cat_cols:
    combined = pd.concat([
        fn_df[col].fillna("不明").rename("val").to_frame().assign(group="FN"),
        non_fn_lost[col].fillna("不明").rename("val").to_frame().assign(group="検出失踪者"),
        non_lost[col].fillna("不明").rename("val").to_frame().assign(group="未失踪者"),
    ])
    ct = pd.crosstab(combined["val"], combined["group"])
    if ct.shape[0] < 2:
        continue
    chi2, p, dof, _ = chi2_contingency(ct)

    sig   = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    sig_c = COLOR_FN if p < 0.05 else "#7f8c8d"

    fn_top   = fn_df[col].fillna("不明").value_counts().index[0]
    det_top  = non_fn_lost[col].fillna("不明").value_counts().index[0]
    safe_top = non_lost[col].fillna("不明").value_counts().index[0]

    rows.append({
        "変数":       col,
        "p値":        p,
        "p値表示":    f"{p:.4f}" if p >= 0.001 else "<0.001",
        "有意":       sig,
        "有意色":     sig_c,
        "共通FN最多":      str(fn_top)[:12],
        "検出失踪者最多":   str(det_top)[:12],
        "未失踪者最多":    str(safe_top)[:12],
    })

# 入国時年齢（Mann-Whitney → Kruskal-Wallis的に追加）
from scipy.stats import kruskal
stat_kw, p_kw = kruskal(
    fn_df["入国時年齢"].dropna(),
    non_fn_lost["入国時年齢"].dropna(),
    non_lost["入国時年齢"].dropna()
)
sig_kw   = "***" if p_kw < 0.001 else "**" if p_kw < 0.01 else "*" if p_kw < 0.05 else "n.s."
sig_kw_c = COLOR_FN if p_kw < 0.05 else "#7f8c8d"
rows.append({
    "変数":       "入国時年齢",
    "p値":        p_kw,
    "p値表示":    f"{p_kw:.4f}" if p_kw >= 0.001 else "<0.001",
    "有意":       sig_kw,
    "有意色":     sig_kw_c,
    "共通FN最多":      f"平均{fn_df['入国時年齢'].mean():.1f}歳",
    "検出失踪者最多":   f"平均{non_fn_lost['入国時年齢'].mean():.1f}歳",
    "未失踪者最多":    f"平均{non_lost['入国時年齢'].mean():.1f}歳",
})

df_result = pd.DataFrame(rows).sort_values("p値")

# =====================================
# 3. 図①：シンプルな横棒グラフ（p値順）
# =====================================
fig, ax = plt.subplots(figsize=(11, 6))

y_pos  = np.arange(len(df_result))
p_vals = df_result["p値"].values
colors_bar = [COLOR_FN if p < 0.05 else "#bdc3c7" for p in p_vals]

bars = ax.barh(y_pos, [-np.log10(p + 1e-10) for p in p_vals],
               color=colors_bar, edgecolor='white', height=0.6)

# 閾値の線
ax.axvline(-np.log10(0.05),  color='#e67e22', linestyle='--',
           linewidth=1.8, label='p = 0.05', zorder=3)
ax.axvline(-np.log10(0.01),  color='#e74c3c', linestyle=':',
           linewidth=1.5, label='p = 0.01', zorder=3)
ax.axvline(-np.log10(0.001), color='#8e44ad', linestyle='-.',
           linewidth=1.5, label='p = 0.001', zorder=3)

# 右端に p値 と 有意記号
for i, row in enumerate(df_result.itertuples()):
    ax.text(-np.log10(row.p値 + 1e-10) + 0.05,
            i, f"  {row.p値表示}  {row.有意}",
            va='center', fontsize=11,
            color=row.有意色, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(df_result["変数"], fontsize=12)
ax.set_xlabel("-log₁₀(p値)　　値が大きいほど3グループ間の差が大きい",
              fontsize=11)
ax.set_title("各属性における3グループ間の差異（カイ二乗検定 / Kruskal-Wallis検定）",
             fontsize=13, fontweight='bold', pad=15)

legend_patches = [
    mpatches.Patch(color=COLOR_FN,   label='有意差あり（p < 0.05）'),
    mpatches.Patch(color='#bdc3c7',  label='有意差なし（n.s.）'),
]
ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0][3:],
          fontsize=10, loc='lower right')

ax.set_xlim(0, max(-np.log10(p_vals + 1e-10)) * 1.35)
ax.grid(axis='x', alpha=0.25, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("PPT3G_07_chi2_clean.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT3G_07_chi2_clean.png 保存完了")

# =====================================
# 4. 図②：テーブル形式まとめ（PPT向け）
# =====================================
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.axis('off')

col_labels = ["属性", "p値", "有意", "共通FN\n最多カテゴリ",
              "検出失踪者\n最多カテゴリ", "未失踪者\n最多カテゴリ"]
table_data = []
row_colors = []

for _, row in df_result.iterrows():
    table_data.append([
        row["変数"],
        row["p値表示"],
        row["有意"],
        row["共通FN最多"],
        row["検出失踪者最多"],
        row["未失踪者最多"],
    ])
    if row["p値"] < 0.001:
        bg = "#fdeaea"
    elif row["p値"] < 0.05:
        bg = "#fef5e7"
    else:
        bg = "#f8f9fa"
    row_colors.append([bg] * 6)

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    cellColours=row_colors
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# ヘッダーのスタイル
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color='white', fontweight='bold')

# 有意列の色
for i, (_, row) in enumerate(df_result.iterrows(), 1):
    cell = table[i, 2]
    if row["p値"] < 0.001:
        cell.set_text_props(color=COLOR_FN, fontweight='bold')
    elif row["p値"] < 0.05:
        cell.set_text_props(color='#e67e22', fontweight='bold')
    else:
        cell.set_text_props(color='#7f8c8d')

# 凡例
legend_text = (
    "■ 背景色：  濃赤 = p<0.001（***）　薄橙 = p<0.05（*/**）　白 = n.s."
)
ax.text(0.5, 0.02, legend_text, transform=ax.transAxes,
        ha='center', fontsize=9.5, color='#555555',
        bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

ax.set_title("カイ二乗検定・Kruskal-Wallis検定 結果まとめ（3グループ比較）",
             fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("PPT3G_07_chi2_table.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT3G_07_chi2_table.png 保存完了")

print("\n✅ 完了！")
print("  PPT3G_07_chi2_clean.png  … シンプルな横棒グラフ版")
print("  PPT3G_07_chi2_table.png  … テーブル形式版")
