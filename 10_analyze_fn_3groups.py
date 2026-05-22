"""
PPT用：共通FN vs 検出できた失踪者 vs 未失踪者 の3グループ比較
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal

# =====================================
# 0. フォント設定（Mac用）
# =====================================
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

COLOR_FN     = "#e74c3c"   # 赤：共通FN
COLOR_DETECT = "#3498db"   # 青：検出できた失踪者
COLOR_SAFE   = "#2ecc71"   # 緑：未失踪者

# =====================================
# 1. データ読み込み・3グループ分け
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

n_fn   = len(fn_df)
n_det  = len(non_fn_lost)
n_safe = len(non_lost)

print(f"共通FN:          {n_fn}件")
print(f"検出できた失踪者: {n_det}件")
print(f"未失踪者:         {n_safe}件")

groups = [
    (fn_df,       f"共通FN\n({n_fn}件)",       COLOR_FN),
    (non_fn_lost, f"検出失踪者\n({n_det}件)",   COLOR_DETECT),
    (non_lost,    f"未失踪者\n({n_safe}件)",     COLOR_SAFE),
]
labels_short = [f"共通FN\n({n_fn}件)",
                f"検出失踪者\n({n_det}件)",
                f"未失踪者\n({n_safe}件)"]

# =====================================
# 図1：3グループの全体サマリー（積み上げ棒）
# =====================================
fig, ax = plt.subplots(figsize=(8, 6))

total = n_fn + n_det + n_safe
sizes = [n_fn, n_det, n_safe]
colors = [COLOR_FN, COLOR_DETECT, COLOR_SAFE]
labels_pie = [
    f"共通FN（{n_fn}件, {n_fn/total*100:.1f}%）",
    f"検出できた失踪者（{n_det}件, {n_det/total*100:.1f}%）",
    f"未失踪者（{n_safe}件, {n_safe/total*100:.1f}%）",
]
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels_pie, colors=colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops=dict(fontsize=10)
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax.set_title("全データ（3041件）の内訳", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("PPT3G_01_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT3G_01_overview.png 保存完了")

# =====================================
# 図2：入国時年齢の3グループ比較
# =====================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("入国時年齢の3グループ比較", fontsize=15, fontweight='bold')

age_fn   = fn_df["入国時年齢"].dropna()
age_det  = non_fn_lost["入国時年齢"].dropna()
age_safe = non_lost["入国時年齢"].dropna()

# Kruskal-Wallis検定（3グループ）
stat_kw, p_kw = kruskal(age_fn, age_det, age_safe)
sig_kw = f"p={p_kw:.3f}" + ("***" if p_kw < 0.001 else "**" if p_kw < 0.01 else "*" if p_kw < 0.05 else "")

# 左：箱ひげ図
ax = axes[0]
bp = ax.boxplot(
    [age_fn, age_det, age_safe],
    labels=labels_short,
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5)
)
for box, color in zip(bp['boxes'], colors):
    box.set_facecolor(color)
    box.set_alpha(0.75)

ax.scatter([1, 2, 3],
           [age_fn.mean(), age_det.mean(), age_safe.mean()],
           color='black', zorder=5, s=80, marker='D', label='平均値')

for i, (data, color) in enumerate(zip([age_fn, age_det, age_safe], colors), 1):
    ax.text(i, data.max() + 0.5, f"平均\n{data.mean():.1f}歳",
            ha='center', fontsize=8, color=color, fontweight='bold')

ax.set_title(f"箱ひげ図\n（Kruskal-Wallis: {sig_kw}）",
             fontsize=11, fontweight='bold')
ax.set_ylabel("入国時年齢（歳）")
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 右：ヒストグラム重ね
ax2 = axes[1]
bins = np.linspace(
    min(age_fn.min(), age_det.min(), age_safe.min()),
    max(age_fn.max(), age_det.max(), age_safe.max()), 18
)
ax2.hist(age_safe, bins=bins, color=COLOR_SAFE,   alpha=0.5,
         label=f"未失踪者（平均{age_safe.mean():.1f}歳）", edgecolor='white')
ax2.hist(age_det,  bins=bins, color=COLOR_DETECT, alpha=0.6,
         label=f"検出失踪者（平均{age_det.mean():.1f}歳）", edgecolor='white')
ax2.hist(age_fn,   bins=bins, color=COLOR_FN,     alpha=0.8,
         label=f"共通FN（平均{age_fn.mean():.1f}歳）", edgecolor='white')
for data, color in zip([age_fn, age_det, age_safe], colors):
    ax2.axvline(data.mean(), color=color, linestyle='--', linewidth=2)
ax2.set_title("年齢分布の重ね合わせ", fontsize=11, fontweight='bold')
ax2.set_xlabel("入国時年齢（歳）")
ax2.set_ylabel("件数")
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("PPT3G_02_age.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT3G_02_age.png 保存完了")

# =====================================
# 図3：職種関係の3グループ比較
# =====================================
fig, ax = plt.subplots(figsize=(13, 6))

col = "職種関係"
fn_vc   = fn_df[col].fillna("不明").value_counts(normalize=True) * 100
det_vc  = non_fn_lost[col].fillna("不明").value_counts(normalize=True) * 100
safe_vc = non_lost[col].fillna("不明").value_counts(normalize=True) * 100
all_cats = sorted(set(fn_vc.index) | set(det_vc.index) | set(safe_vc.index))

x = np.arange(len(all_cats))
w = 0.25
for i, (vc, label, color) in enumerate(zip(
    [fn_vc, det_vc, safe_vc],
    [f"共通FN({n_fn}件)", f"検出失踪者({n_det}件)", f"未失踪者({n_safe}件)"],
    colors
)):
    vals = [vc.get(c, 0) for c in all_cats]
    bars = ax.bar(x + (i-1)*w, vals, w, label=label,
                  color=color, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        if val > 2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.0f}%", ha='center', va='bottom',
                    fontsize=8, color=color, fontweight='bold')

ax.set_title("職種関係の割合比較（3グループ）", fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(all_cats, rotation=20, ha='right', fontsize=10)
ax.set_ylabel("割合（%）")
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("PPT3G_03_shokushu_kankei.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT3G_03_shokushu_kankei.png 保存完了")

# =====================================
# 図4：職種（具体）の3グループ上位比較
# =====================================
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("職種（具体）の上位8件比較", fontsize=15, fontweight='bold')

col = "職種"
for ax, (df_plot, label, color) in zip(axes, groups):
    vc = df_plot[col].fillna("不明").value_counts().head(8)
    bars = ax.barh(vc.index[::-1], vc.values[::-1],
                   color=color, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vc.values[::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=9, fontweight='bold')
    ax.set_title(label.replace('\n', ' '), fontsize=11,
                 fontweight='bold', color=color)
    ax.set_xlabel("件数")
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("PPT3G_04_shokushu_detail.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT3G_04_shokushu_detail.png 保存完了")

# =====================================
# 図5：都道府県の3グループ比較
# =====================================
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("実習先都道府県の上位8件比較", fontsize=15, fontweight='bold')

col = "所在地(実習先)(都道府県)"
for ax, (df_plot, label, color) in zip(axes, groups):
    vc = df_plot[col].fillna("不明").value_counts().head(8)
    bars = ax.barh(vc.index[::-1], vc.values[::-1],
                   color=color, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vc.values[::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=9, fontweight='bold')
    ax.set_title(label.replace('\n', ' '), fontsize=11,
                 fontweight='bold', color=color)
    ax.set_xlabel("件数")
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("PPT3G_05_prefecture.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT3G_05_prefecture.png 保存完了")

# =====================================
# 図6：性別の3グループ比較
# =====================================
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle("性別の3グループ比較", fontsize=15, fontweight='bold')

col = "性別"
for ax, (df_plot, label, color) in zip(axes, groups):
    vc = df_plot[col].fillna("不明").value_counts()
    total_g = vc.sum()
    ax.pie(vc.values, labels=vc.index,
           colors=[color, "#bdc3c7"],
           autopct='%1.1f%%', startangle=90,
           wedgeprops=dict(edgecolor='white', linewidth=2))
    ax.set_title(label.replace('\n', ' '), fontsize=11,
                 fontweight='bold', color=color)

plt.tight_layout()
plt.savefig("PPT3G_06_gender.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT3G_06_gender.png 保存完了")

# =====================================
# 図7：カイ二乗検定まとめ（3グループ版）
# =====================================
cat_cols = ["性別", "職種関係", "職種", "所在地(実習先)(都道府県)",
            "派遣会社", "学校所属", "組合", "所属機関"]
cat_cols = [c for c in cat_cols if c in fn_df.columns]

chi2_rows = []
for col in cat_cols:
    combined = pd.concat([
        fn_df[col].fillna("不明").rename("val").to_frame().assign(group="共通FN"),
        non_fn_lost[col].fillna("不明").rename("val").to_frame().assign(group="検出失踪者"),
        non_lost[col].fillna("不明").rename("val").to_frame().assign(group="未失踪者"),
    ])
    ct = pd.crosstab(combined["val"], combined["group"])
    if ct.shape[0] < 2:
        continue
    chi2, p, dof, _ = chi2_contingency(ct)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    chi2_rows.append({"変数": col, "p値": p, "有意": sig,
                      "-log10(p)": -np.log10(p + 1e-10)})

chi2_3g = pd.DataFrame(chi2_rows).sort_values("-log10(p)", ascending=True)

fig, ax = plt.subplots(figsize=(12, 6))
bar_colors = [COLOR_FN if p < 0.05 else "#bdc3c7" for p in chi2_3g["p値"]]
bars = ax.barh(chi2_3g["変数"], chi2_3g["-log10(p)"],
               color=bar_colors, alpha=0.85, edgecolor='white')
ax.axvline(-np.log10(0.05),  color='orange', linestyle='--', linewidth=2, label='p=0.05')
ax.axvline(-np.log10(0.001), color='red',    linestyle='--', linewidth=2, label='p=0.001')

for bar, row in zip(bars, chi2_3g.itertuples()):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            row.有意, va='center', fontsize=11, fontweight='bold',
            color=COLOR_FN if row.p値 < 0.05 else 'gray')

ax.set_xlabel("-log10(p値)　値が大きいほど3グループ間の差が大きい", fontsize=10)
ax.set_title("各属性のカイ二乗検定結果（3グループ比較）\n（赤：p<0.05で有意）",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

chi2_3g.to_csv("chi2_3groups.csv", index=False, encoding="utf-8-sig")

plt.tight_layout()
plt.savefig("PPT3G_07_chi2_3groups.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT3G_07_chi2_3groups.png 保存完了")

# =====================================
# 図8：組合・所属機関の3グループ比較
# =====================================
for col, fname in [("組合", "PPT3G_08_union.png"),
                   ("所属機関", "PPT3G_09_organization.png")]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(f"{col}の上位5件比較（3グループ）", fontsize=15, fontweight='bold')
    for ax, (df_plot, label, color) in zip(axes, groups):
        vc = df_plot[col].fillna("不明").value_counts().head(5)
        bars = ax.barh(vc.index[::-1], vc.values[::-1],
                       color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vc.values[::-1]):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', fontsize=9, fontweight='bold')
        ax.set_title(label.replace('\n', ' '), fontsize=11,
                     fontweight='bold', color=color)
        ax.set_xlabel("件数")
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"→ {fname} 保存完了")

print("\n✅ 全図の生成完了！")
print("\n生成されたPPT用ファイル一覧（3グループ比較）:")
print("  PPT3G_01_overview.png          … 全体サマリー円グラフ")
print("  PPT3G_02_age.png               … 入国時年齢の3グループ比較")
print("  PPT3G_03_shokushu_kankei.png   … 職種関係の割合比較")
print("  PPT3G_04_shokushu_detail.png   … 職種（具体）上位8件")
print("  PPT3G_05_prefecture.png        … 都道府県の比較")
print("  PPT3G_06_gender.png            … 性別の比較")
print("  PPT3G_07_chi2_3groups.png      … カイ二乗検定まとめ")
print("  PPT3G_08_union.png             … 組合の比較")
print("  PPT3G_09_organization.png      … 所属機関の比較")
