"""
PPT用：共通FN（29件）vs 検出できた失踪者 の比較可視化スクリプト
各図を個別のPNGファイルとして保存する
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import chi2_contingency, mannwhitneyu
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# =====================================
# 0. フォント設定（Mac用）
# =====================================
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# カラー設定
COLOR_FN     = "#e74c3c"   # 赤：共通FN
COLOR_DETECT = "#3498db"   # 青：検出できた失踪者
COLOR_ALL    = "#95a5a6"   # グレー：全体

# =====================================
# 1. データ読み込み・準備
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

fn_keys   = set(make_key(fn_df, KEY_COLS))
lost_keys = make_key(lost_all, KEY_COLS)
non_fn_lost = lost_all[~lost_keys.isin(fn_keys)].copy()

print(f"共通FN:          {len(fn_df)}件")
print(f"検出できた失踪者: {len(non_fn_lost)}件")
print(f"失踪者合計:       {len(lost_all)}件")

# =====================================
# 図1：全体のサマリー（円グラフ + 概要）
# =====================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("失踪者の検出状況サマリー", fontsize=16, fontweight='bold', y=1.02)

# 左：円グラフ（全データの構成）
ax1 = axes[0]
sizes_all = [len(non_lost), len(non_fn_lost), len(fn_df)]
labels_all = [
    f"非失踪者\n({len(non_lost)}件)",
    f"検出できた失踪者\n({len(non_fn_lost)}件)",
    f"共通FN\n({len(fn_df)}件)"
]
colors_all = [COLOR_ALL, COLOR_DETECT, COLOR_FN]
wedges, texts, autotexts = ax1.pie(
    sizes_all, labels=labels_all, colors=colors_all,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops=dict(fontsize=10)
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight('bold')
ax1.set_title("全データの構成", fontsize=13, fontweight='bold')

# 右：失踪者内の円グラフ
ax2 = axes[1]
sizes_lost = [len(non_fn_lost), len(fn_df)]
labels_lost = [
    f"検出できた失踪者\n({len(non_fn_lost)}件, {len(non_fn_lost)/len(lost_all)*100:.1f}%)",
    f"全モデル共通FN\n({len(fn_df)}件, {len(fn_df)/len(lost_all)*100:.1f}%)"
]
wedges2, texts2, autotexts2 = ax2.pie(
    sizes_lost, labels=labels_lost, colors=[COLOR_DETECT, COLOR_FN],
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops=dict(fontsize=10)
)
for at in autotexts2:
    at.set_fontsize(11)
    at.set_fontweight('bold')
ax2.set_title("失踪者102件の内訳", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("PPT_01_summary_pie.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT_01_summary_pie.png 保存完了")

# =====================================
# 図2：入国時年齢の比較（箱ひげ図 + バイオリン図）
# =====================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("入国時年齢の比較", fontsize=16, fontweight='bold')

data_fn    = fn_df["入国時年齢"].dropna()
data_det   = non_fn_lost["入国時年齢"].dropna()
stat, p    = mannwhitneyu(data_fn, data_det, alternative='two-sided')
sig_label  = f"p={p:.3f}" + ("*" if p < 0.05 else "")

# 左：箱ひげ図
ax = axes[0]
bp = ax.boxplot(
    [data_fn, data_det],
    labels=["共通FN\n(29件)", "検出できた\n失踪者(73件)"],
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5)
)
bp['boxes'][0].set_facecolor(COLOR_FN);    bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(COLOR_DETECT); bp['boxes'][1].set_alpha(0.7)
ax.scatter([1, 2], [data_fn.mean(), data_det.mean()],
           color='black', zorder=5, s=80, marker='D', label='平均値')
ax.set_title(f"箱ひげ図（Mann-Whitney U検定: {sig_label}）",
             fontsize=11, fontweight='bold')
ax.set_ylabel("入国時年齢（歳）")
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# 右：ヒストグラム重ね
ax2 = axes[1]
bins = np.linspace(
    min(data_fn.min(), data_det.min()),
    max(data_fn.max(), data_det.max()), 15
)
ax2.hist(data_det, bins=bins, color=COLOR_DETECT, alpha=0.6,
         label=f"検出できた失踪者（平均{data_det.mean():.1f}歳）", edgecolor='white')
ax2.hist(data_fn, bins=bins, color=COLOR_FN, alpha=0.7,
         label=f"共通FN（平均{data_fn.mean():.1f}歳）", edgecolor='white')
ax2.axvline(data_fn.mean(),  color=COLOR_FN,     linestyle='--', linewidth=2)
ax2.axvline(data_det.mean(), color=COLOR_DETECT,  linestyle='--', linewidth=2)
ax2.set_title("年齢分布の重ね合わせ", fontsize=11, fontweight='bold')
ax2.set_xlabel("入国時年齢（歳）")
ax2.set_ylabel("件数")
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("PPT_02_age_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT_02_age_comparison.png 保存完了")

# =====================================
# 図3：職種関係の比較（横並び棒グラフ）
# =====================================
fig, ax = plt.subplots(figsize=(12, 6))

col = "職種関係"
fn_vc    = fn_df[col].fillna("不明").value_counts(normalize=True) * 100
det_vc   = non_fn_lost[col].fillna("不明").value_counts(normalize=True) * 100
all_cats = sorted(set(fn_vc.index) | set(det_vc.index))
fn_vals  = [fn_vc.get(c, 0) for c in all_cats]
det_vals = [det_vc.get(c, 0) for c in all_cats]

x = np.arange(len(all_cats))
w = 0.35
bars1 = ax.bar(x - w/2, fn_vals,  w, label="共通FN（29件）",       color=COLOR_FN,     alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + w/2, det_vals, w, label="検出できた失踪者（73件）", color=COLOR_DETECT, alpha=0.85, edgecolor='white')

for bar, val in zip(bars1, fn_vals):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha='center', va='bottom', fontsize=9, color=COLOR_FN, fontweight='bold')
for bar, val in zip(bars2, det_vals):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha='center', va='bottom', fontsize=9, color=COLOR_DETECT, fontweight='bold')

ax.set_title("職種関係の割合比較（カイ二乗検定: p<0.001***）",
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(all_cats, rotation=20, ha='right', fontsize=10)
ax.set_ylabel("割合（%）")
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(max(fn_vals), max(det_vals)) * 1.2)

plt.tight_layout()
plt.savefig("PPT_03_shokushu_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT_03_shokushu_comparison.png 保存完了")

# =====================================
# 図4：職種（具体的）の比較（上位8件）
# =====================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("職種（具体）の分布比較", fontsize=15, fontweight='bold')

col = "職種"
for ax, (df_plot, label, color) in zip(axes, [
    (fn_df,       "共通FN（29件）",       COLOR_FN),
    (non_fn_lost, "検出できた失踪者（73件）", COLOR_DETECT)
]):
    vc = df_plot[col].fillna("不明").value_counts().head(8)
    bars = ax.barh(vc.index[::-1], vc.values[::-1],
                   color=color, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vc.values[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=10, fontweight='bold')
    ax.set_title(label, fontsize=12, fontweight='bold', color=color)
    ax.set_xlabel("件数")
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("PPT_04_shokushu_detail.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT_04_shokushu_detail.png 保存完了")

# =====================================
# 図5：都道府県の比較（上位8件）
# =====================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("実習先都道府県の分布比較", fontsize=15, fontweight='bold')

col = "所在地(実習先)(都道府県)"
for ax, (df_plot, label, color) in zip(axes, [
    (fn_df,       "共通FN（29件）",       COLOR_FN),
    (non_fn_lost, "検出できた失踪者（73件）", COLOR_DETECT)
]):
    vc = df_plot[col].fillna("不明").value_counts().head(8)
    bars = ax.barh(vc.index[::-1], vc.values[::-1],
                   color=color, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vc.values[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=10, fontweight='bold')
    ax.set_title(label, fontsize=12, fontweight='bold', color=color)
    ax.set_xlabel("件数")
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("PPT_05_prefecture_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT_05_prefecture_comparison.png 保存完了")

# =====================================
# 図6：組合・所属機関の比較（上位5件）
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("組合・所属機関の分布比較", fontsize=15, fontweight='bold')

for row, col in enumerate(["組合", "所属機関"]):
    for col_idx, (df_plot, label, color) in enumerate([
        (fn_df,       "共通FN（29件）",       COLOR_FN),
        (non_fn_lost, "検出できた失踪者（73件）", COLOR_DETECT)
    ]):
        ax = axes[row][col_idx]
        vc = df_plot[col].fillna("不明").value_counts().head(5)
        bars = ax.barh(vc.index[::-1], vc.values[::-1],
                       color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vc.values[::-1]):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', fontsize=9, fontweight='bold')
        ax.set_title(f"{col}：{label}", fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel("件数")
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("PPT_06_union_organization.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT_06_union_organization.png 保存完了")

# =====================================
# 図7：カイ二乗検定の結果まとめ（横棒グラフ）
# =====================================
cat_cols = ["性別", "職種関係", "職種", "所在地(実習先)(都道府県)",
            "派遣会社", "学校所属", "組合", "所属機関"]
cat_cols = [c for c in cat_cols if c in fn_df.columns and c in non_fn_lost.columns]

chi2_results = []
for col in cat_cols:
    combined = pd.concat([
        fn_df[col].fillna("不明").rename("val").to_frame().assign(group="FN"),
        non_fn_lost[col].fillna("不明").rename("val").to_frame().assign(group="非FN")
    ])
    ct = pd.crosstab(combined["val"], combined["group"])
    if ct.shape[0] < 2:
        continue
    chi2, p, dof, _ = chi2_contingency(ct)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    chi2_results.append({"変数": col, "p値": p, "有意": sig, "-log10(p)": -np.log10(p + 1e-10)})

chi2_df = pd.DataFrame(chi2_results).sort_values("-log10(p)", ascending=True)

fig, ax = plt.subplots(figsize=(12, 6))
colors_bar = [COLOR_FN if p < 0.05 else COLOR_ALL for p in chi2_df["p値"]]
bars = ax.barh(chi2_df["変数"], chi2_df["-log10(p)"],
               color=colors_bar, alpha=0.85, edgecolor='white')

ax.axvline(-np.log10(0.05),  color='orange', linestyle='--',
           linewidth=2, label='p=0.05')
ax.axvline(-np.log10(0.001), color='red',    linestyle='--',
           linewidth=2, label='p=0.001')

for bar, row in zip(bars, chi2_df.itertuples()):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            row.有意, va='center', fontsize=11, fontweight='bold',
            color=COLOR_FN if row.p値 < 0.05 else 'gray')

ax.set_xlabel("-log10(p値)　値が大きいほど有意差が強い", fontsize=11)
ax.set_title("各属性のカイ二乗検定結果\n（赤：p<0.05で有意、グレー：有意差なし）",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("PPT_07_chi2_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT_07_chi2_summary.png 保存完了")

# =====================================
# 図8：全属性の割合比較（ヒートマップ）
# =====================================
fig, axes = plt.subplots(1, len(cat_cols[:4]), figsize=(18, 7))
fig.suptitle("各属性の割合比較（共通FN vs 検出できた失踪者）上位5件",
             fontsize=14, fontweight='bold')

for ax, col in zip(axes, cat_cols[:4]):
    fn_vc  = fn_df[col].fillna("不明").value_counts(normalize=True).head(5) * 100
    det_vc = non_fn_lost[col].fillna("不明").value_counts(normalize=True).head(5) * 100
    top_cats = list(fn_vc.index[:5])

    fn_vals_  = [fn_vc.get(c, 0) for c in top_cats]
    det_vals_ = [det_vc.get(c, 0) for c in top_cats]

    x_ = np.arange(len(top_cats))
    w_ = 0.35
    ax.bar(x_ - w_/2, fn_vals_,  w_, color=COLOR_FN,     alpha=0.8, label="共通FN")
    ax.bar(x_ + w_/2, det_vals_, w_, color=COLOR_DETECT,  alpha=0.8, label="検出できた失踪者")
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.set_xticks(x_)
    ax.set_xticklabels(top_cats, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("%")
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("PPT_08_multi_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ PPT_08_multi_comparison.png 保存完了")

print("\n✅ 全図の生成完了！")
print("\n生成されたPPT用ファイル一覧:")
print("  PPT_01_summary_pie.png        … 全体サマリー（円グラフ）")
print("  PPT_02_age_comparison.png     … 入国時年齢の比較")
print("  PPT_03_shokushu_comparison.png … 職種関係の割合比較")
print("  PPT_04_shokushu_detail.png    … 職種（具体）の比較")
print("  PPT_05_prefecture_comparison.png … 都道府県の比較")
print("  PPT_06_union_organization.png … 組合・所属機関の比較")
print("  PPT_07_chi2_summary.png       … カイ二乗検定まとめ")
print("  PPT_08_multi_comparison.png   … 複数属性の割合比較")
