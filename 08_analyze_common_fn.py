"""
全モデル共通FN（29件）の詳細分析スクリプト
① 属性分布の可視化
② 非FN失踪者（他の失踪者）との比較
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import chi2_contingency, mannwhitneyu

# =====================================
# 0. フォント設定（Mac用）
# =====================================
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# =====================================
# 1. データ読み込み
# =====================================
fn_df   = pd.read_csv("Common_FN_all_models_2.csv")   # 29件：全モデル共通FN
all_df  = pd.read_csv("clean2.csv")                    # 全データ

# 失踪者のみ抽出
lost_all = all_df[all_df["失踪の有無"] == 1].copy()

# FNのキーを作成して非FN失踪者を分離
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

fn_keys   = set(make_key(fn_df,   KEY_COLS))
lost_keys = make_key(lost_all, KEY_COLS)

# 非FN失踪者：他のモデルで検出できた失踪者
non_fn_lost = lost_all[~lost_keys.isin(fn_keys)].copy()

fn_df["グループ"]      = "共通FN（29件）"
non_fn_lost["グループ"] = "検出できた失踪者"

print(f"共通FN件数:        {len(fn_df)}件")
print(f"検出できた失踪者:  {len(non_fn_lost)}件")
print(f"失踪者合計:        {len(lost_all)}件")

# =====================================
# 2. ① 属性分布の可視化（FN29件のみ）
# =====================================
cat_cols = ["性別", "職種関係", "職種", "所在地(実習先)(都道府県)",
            "派遣会社", "学校所属", "組合", "所属機関"]
cat_cols = [c for c in cat_cols if c in fn_df.columns]

# 各カテゴリ列について上位8件を棒グラフ化
n_cols = len(cat_cols)
ncols_plot = 2
nrows_plot = (n_cols + 1) // ncols_plot

fig, axes = plt.subplots(nrows_plot, ncols_plot,
                         figsize=(16, nrows_plot * 4))
fig.suptitle("共通FN（29件）の属性分布", fontsize=15, fontweight='bold')

for i, col in enumerate(cat_cols):
    ax = axes.flatten()[i]
    vc = fn_df[col].fillna("不明").value_counts().head(8)
    bars = ax.barh(vc.index[::-1], vc.values[::-1],
                   color="#e74c3c", alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, vc.values[::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=9)
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_xlabel("件数")
    ax.grid(axis='x', alpha=0.3)

# 入国時年齢はヒストグラム
if "入国時年齢" in fn_df.columns:
    ax_age = axes.flatten()[n_cols] if n_cols < len(axes.flatten()) else None
    if ax_age:
        ax_age.hist(fn_df["入国時年齢"].dropna(), bins=10,
                    color="#e74c3c", alpha=0.8, edgecolor='white')
        ax_age.set_title("入国時年齢の分布", fontsize=11, fontweight='bold')
        ax_age.set_xlabel("年齢")
        ax_age.set_ylabel("件数")
        ax_age.grid(axis='y', alpha=0.3)

# 余った枠を非表示
for j in range(n_cols + 1, len(axes.flatten())):
    axes.flatten()[j].axis('off')

plt.tight_layout()
plt.savefig("CommonFN_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n→ CommonFN_distribution.png 保存完了")

# =====================================
# 3. ② 非FN失踪者との比較
# =====================================

# --- 3-1. 入国時年齢の比較（箱ひげ図 + 統計検定）---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("共通FNと検出できた失踪者の属性比較", fontsize=14, fontweight='bold')

ax = axes[0]
data_fn     = fn_df["入国時年齢"].dropna()
data_nonfn  = non_fn_lost["入国時年齢"].dropna()

bp = ax.boxplot([data_fn, data_nonfn],
                labels=["共通FN\n(29件)", "検出できた\n失踪者"],
                patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor("#e74c3c")
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor("#3498db")
bp['boxes'][1].set_alpha(0.7)

# Mann-Whitney U検定
stat, p = mannwhitneyu(data_fn, data_nonfn, alternative='two-sided')
ax.set_title(f"入国時年齢の比較\n(Mann-Whitney U, p={p:.3f}{'*' if p<0.05 else ''})",
             fontsize=11, fontweight='bold')
ax.set_ylabel("入国時年齢")
ax.grid(axis='y', alpha=0.3)

# 平均値を点で追加
ax.scatter([1, 2],
           [data_fn.mean(), data_nonfn.mean()],
           color='black', zorder=5, s=50, marker='D', label='平均値')
ax.legend(fontsize=9)

# --- 3-2. カテゴリ変数の比較（職種関係） ---
ax2 = axes[1]
compare_col = "職種関係" if "職種関係" in fn_df.columns else cat_cols[0]

fn_vc     = fn_df[compare_col].fillna("不明").value_counts(normalize=True) * 100
nonfn_vc  = non_fn_lost[compare_col].fillna("不明").value_counts(normalize=True) * 100

all_cats = sorted(set(fn_vc.index) | set(nonfn_vc.index))
fn_vals    = [fn_vc.get(c, 0) for c in all_cats]
nonfn_vals = [nonfn_vc.get(c, 0) for c in all_cats]

x = np.arange(len(all_cats))
width = 0.35
bars1 = ax2.bar(x - width/2, fn_vals,    width, label="共通FN",        color="#e74c3c", alpha=0.8)
bars2 = ax2.bar(x + width/2, nonfn_vals, width, label="検出できた失踪者", color="#3498db", alpha=0.8)

ax2.set_title(f"{compare_col}の割合比較（%）", fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(all_cats, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel("割合（%）")
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("CommonFN_vs_detected.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ CommonFN_vs_detected.png 保存完了")

# =====================================
# 4. 全カテゴリ変数の比較（カイ二乗検定）
# =====================================
print("\n=== カテゴリ変数のカイ二乗検定（共通FN vs 検出できた失踪者）===")
print(f"{'変数名':20s}  {'p値':>8s}  {'有意':>4s}  {'FN上位値':>20s}  {'非FN上位値':>20s}")
print("-" * 80)

chi2_results = []
for col in cat_cols:
    if col not in non_fn_lost.columns:
        continue
    fn_top    = fn_df[col].fillna("不明").value_counts().index[0]
    nonfn_top = non_fn_lost[col].fillna("不明").value_counts().index[0]

    # 分割表を作成
    combined = pd.concat([
        fn_df[col].fillna("不明").rename("val").to_frame().assign(group="FN"),
        non_fn_lost[col].fillna("不明").rename("val").to_frame().assign(group="非FN")
    ])
    ct = pd.crosstab(combined["val"], combined["group"])
    if ct.shape[0] < 2:
        continue
    chi2, p, dof, _ = chi2_contingency(ct)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    chi2_results.append({
        "変数": col, "chi2": chi2, "p値": p, "有意": sig,
        "FN上位値": fn_top, "非FN上位値": nonfn_top
    })
    print(f"{col:20s}  {p:8.4f}  {sig:>4s}  {str(fn_top)[:20]:>20s}  {str(nonfn_top)[:20]:>20s}")

# 入国時年齢
if "入国時年齢" in fn_df.columns:
    stat, p = mannwhitneyu(
        fn_df["入国時年齢"].dropna(),
        non_fn_lost["入国時年齢"].dropna(),
        alternative='two-sided'
    )
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    fn_mean    = fn_df["入国時年齢"].mean()
    nonfn_mean = non_fn_lost["入国時年齢"].mean()
    print(f"{'入国時年齢':20s}  {p:8.4f}  {sig:>4s}  {'平均'+str(round(fn_mean,1))+'歳':>20s}  {'平均'+str(round(nonfn_mean,1))+'歳':>20s}")

# =====================================
# 5. 結果をCSVに保存
# =====================================
chi2_df = pd.DataFrame(chi2_results)
chi2_df.to_csv("CommonFN_chi2_results.csv", index=False, encoding="utf-8-sig")
print("\n→ CommonFN_chi2_results.csv 保存完了")

print("\n✅ 分析完了！")
print("\n生成ファイル:")
print("  CommonFN_distribution.png   … 29件の属性分布")
print("  CommonFN_vs_detected.png    … FN vs 検出できた失踪者の比較")
print("  CommonFN_chi2_results.csv   … カイ二乗検定の結果一覧")
