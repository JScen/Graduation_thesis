"""
FN重叠/差异分析脚本
各モデル間のFalse Negative（偽陰性）の重複・差異を比較する
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib_venn import venn2, venn3
from itertools import combinations

# =====================================
# 0. 日本語フォント設定（Mac用）
# =====================================
import matplotlib.font_manager as fm

font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
matplotlib.rcParams["font.family"] = prop.get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

# =====================================
# 1. データ読み込み
# =====================================
dt_fn  = pd.read_csv("LOOCV_FN_DecisionTree_2.csv")
rf_fn  = pd.read_csv("LOOCV_FN_RandomForest_2.csv")
lr_fn  = pd.read_csv("LOOCV_FN_LogisticRegression_2.csv")
svm_fn = pd.read_csv("LOOCV_FN_SVM_2.csv")

models = {
    "DecisionTree":        dt_fn,
    "RandomForest":        rf_fn,
    "LogisticRegression":  lr_fn,
    "SVM":                 svm_fn,
}

print("=== 各モデルのFN件数 ===")
for name, df in models.items():
    print(f"  {name}: {len(df)}件")

# =====================================
# 2. キー列の設定（LOOCV_FN2.pyと同じ方式）
# =====================================
KEY_COLS_ALL = [
    "入国時年齢", "性別", "職種関係", "職種",
    "所在地(実習先)(都道府県)", "所在地(実習先)(市区町村)",
    "派遣会社", "学校所属", "組合", "所属機関"
]

# 全モデルで共通して存在する列のみ使用
def common_key_cols(dfs, cand_cols):
    cols = set(cand_cols)
    for df in dfs:
        cols &= set(df.columns)
    return [c for c in cand_cols if c in cols]

KEY_COLS = common_key_cols(list(models.values()), KEY_COLS_ALL)
print(f"\n使用キー列: {KEY_COLS}")

def build_key_set(df, key_cols):
    sub = df[key_cols].copy()
    for c in key_cols:
        sub[c] = sub[c].astype(str).str.strip().replace({"nan": ""})
    return set(sub.agg("||".join, axis=1))

key_sets = {name: build_key_set(df, KEY_COLS) for name, df in models.items()}

# =====================================
# 3. 基本統計：重複数マトリクス
# =====================================
model_names = list(key_sets.keys())
n = len(model_names)

print("\n=== 2モデル間の共通FN件数マトリクス ===")
print(f"{'':20s}", end="")
for name in model_names:
    print(f"{name[:6]:>8s}", end="")
print()

overlap_matrix = {}
for name1 in model_names:
    overlap_matrix[name1] = {}
    print(f"{name1[:20]:20s}", end="")
    for name2 in model_names:
        count = len(key_sets[name1] & key_sets[name2])
        overlap_matrix[name1][name2] = count
        print(f"{count:8d}", end="")
    print()

# =====================================
# 4. 各モデル固有のFN（他のどのモデルでも検出できていないFN）
# =====================================
print("\n=== 各モデル固有のFN（他3モデルでは正しく予測できているFN） ===")
unique_fns = {}
for name, keys in key_sets.items():
    other_keys = set()
    for other_name, other_keys_set in key_sets.items():
        if other_name != name:
            other_keys |= other_keys_set
    unique = keys - other_keys
    unique_fns[name] = unique
    print(f"  {name}: {len(unique)}件")

# =====================================
# 5. 全モデル共通FN
# =====================================
all_common = key_sets["DecisionTree"]
for keys in key_sets.values():
    all_common &= keys
print(f"\n=== 全モデル共通FN: {len(all_common)}件 ===")

# =====================================
# 6. 可視化① BarChart：モデル別FN件数と内訳
# =====================================
# 各モデルのFNを「共通」「2-3モデル共通」「固有」に分類
categories = {}
for name, keys in key_sets.items():
    others = [key_sets[n] for n in model_names if n != name]
    in_all    = len(keys & all_common)
    in_some   = len(keys & (others[0] | others[1] | others[2])) - in_all
    unique_n  = len(unique_fns[name])
    categories[name] = {
        "全モデル共通": in_all,
        "一部モデル共通": in_some,
        "固有FN": unique_n,
    }

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左：FN件数の積み上げバー
cat_labels = ["全モデル共通", "一部モデル共通", "固有FN"]
colors = ["#e74c3c", "#e67e22", "#3498db"]
bottoms = [0] * n
bar_data = {cat: [categories[m][cat] for m in model_names] for cat in cat_labels}

ax = axes[0]
for i, (cat, color) in enumerate(zip(cat_labels, colors)):
    bars = ax.bar(model_names, bar_data[cat], bottom=bottoms, color=color, label=cat, alpha=0.85)
    # 数値ラベル
    for j, (bar, val) in enumerate(zip(bars, bar_data[cat])):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bottoms[j] + val/2,
                    str(val), ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    bottoms = [bottoms[j] + bar_data[cat][j] for j in range(n)]

ax.set_title("FN件数の内訳（モデル別）", fontsize=13, fontweight='bold')
ax.set_ylabel("FN件数")
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

# 右：重複マトリクスのヒートマップ
ax2 = axes[1]
import numpy as np
matrix_vals = np.array([[overlap_matrix[r][c] for c in model_names] for r in model_names])
im = ax2.imshow(matrix_vals, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, ax=ax2, shrink=0.8)

for i in range(n):
    for j in range(n):
        ax2.text(j, i, str(matrix_vals[i, j]),
                 ha='center', va='center',
                 fontsize=12, fontweight='bold',
                 color='white' if matrix_vals[i, j] > matrix_vals.max()*0.6 else 'black')

ax2.set_xticks(range(n))
ax2.set_yticks(range(n))
ax2.set_xticklabels(model_names, rotation=15, ha='right')
ax2.set_yticklabels(model_names)
ax2.set_title("モデル間FN重複数マトリクス", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("FN_overlap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n→ FN_overlap_summary.png 保存完了")

# =====================================
# 7. 可視化② Venn図（3モデルの組み合わせ）
# =====================================
from matplotlib_venn import venn3, venn3_circles
from matplotlib.patches import Patch

short = {"DecisionTree": "DT", "RandomForest": "RF",
         "LogisticRegression": "LR", "SVM": "SVM"}

# 各モデルに固定の色を割り当て
model_colors = {
    "DecisionTree":       "#e74c3c",   # 赤
    "RandomForest":       "#3498db",   # 青
    "LogisticRegression": "#2ecc71",   # 緑
    "SVM":                "#f39c12",   # オレンジ
}

combos = list(combinations(model_names, 3))

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("モデル間FN重複のVenn図", fontsize=15, fontweight='bold', y=1.01)

for ax, (a, b, c) in zip(axes.flatten()[:4], combos):
    sa, sb, sc = key_sets[a], key_sets[b], key_sets[c]

    ca = model_colors[a]
    cb = model_colors[b]
    cc = model_colors[c]

    v = venn3(
        [sa, sb, sc],
        set_labels=("", "", ""),   # ラベルは手動で追加
        set_colors=(ca, cb, cc),
        alpha=0.5,
        ax=ax
    )

    # 各領域のフォントサイズを調整
    for text in v.subset_labels:
        if text:
            text.set_fontsize(11)
            text.set_fontweight('bold')

    # モデル名を円の外側に手動配置（色付き）
    label_positions = [(-0.55, 0.55), (0.55, 0.55), (0.0, -0.65)]
    for i, (name, pos) in enumerate(zip([a, b, c], label_positions)):
        ax.text(pos[0], pos[1], short[name],
                fontsize=13, fontweight='bold',
                color=model_colors[name],
                ha='center', va='center',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white',
                          edgecolor=model_colors[name],
                          linewidth=2))

    ax.set_title(f"{short[a]} / {short[b]} / {short[c]}", fontsize=12, fontweight='bold', pad=10)

# 5枠目：カラー凡例 + 全体サマリー
ax_legend = axes.flatten()[4]
ax_legend.axis('off')

legend_elements = [
    Patch(facecolor=model_colors[name], edgecolor='gray',
          alpha=0.7, label=f"{short[name]}  ({name})")
    for name in model_names
]
ax_legend.legend(
    handles=legend_elements,
    loc='upper center',
    fontsize=12,
    title=" ",
    title_fontsize=12,
    frameon=True,
    framealpha=0.9,
    edgecolor='gray',
    bbox_to_anchor=(0.5, 0.85)
)

# 全体サマリーテキスト
summary_lines = [
    f"全モデル共通FN: {len(all_common)}件",
    "",
    "各モデル固有FN:",
]
for name in model_names:
    summary_lines.append(f"  {short[name]}: {len(unique_fns[name])}件")

ax_legend.text(0.5, 0.35, "\n".join(summary_lines),
               transform=ax_legend.transAxes,
               fontsize=11, va='top', ha='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow',
                         edgecolor='gray', alpha=0.9))

# 6枠目：4モデル全体のFN件数バー（参考）
ax_bar = axes.flatten()[5]
fn_counts = [len(key_sets[m]) for m in model_names]
bar_colors = [model_colors[m] for m in model_names]
bars = ax_bar.bar([short[m] for m in model_names], fn_counts,
                  color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, fn_counts):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom',
                fontsize=12, fontweight='bold')
ax_bar.set_title("各モデルのFN総数", fontsize=12, fontweight='bold')
ax_bar.set_ylabel("FN件数")
ax_bar.grid(axis='y', alpha=0.3)
ax_bar.set_ylim(0, max(fn_counts) * 1.2)

plt.tight_layout()
plt.savefig("FN_venn_diagrams.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ FN_venn_diagrams.png 保存完了")

# =====================================
# 8. 結果CSVの出力
# =====================================
# 各モデル固有のFNデータを保存
for name, unique_keys in unique_fns.items():
    df = models[name].copy()
    sub = df[KEY_COLS].copy()
    for c in KEY_COLS:
        sub[c] = sub[c].astype(str).str.strip().replace({"nan": ""})
    key_col = sub.agg("||".join, axis=1)
    unique_df = df[key_col.isin(unique_keys)]
    fname = f"FN_unique_{name}.csv"
    unique_df.to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"→ {fname} 保存完了（{len(unique_df)}件）")

# 重複マトリクスをCSVとして保存
overlap_df = pd.DataFrame(overlap_matrix).loc[model_names, model_names]
overlap_df.to_csv("FN_overlap_matrix.csv", encoding="utf-8-sig")
print("→ FN_overlap_matrix.csv 保存完了")

print("\n✅ 分析完了！")
