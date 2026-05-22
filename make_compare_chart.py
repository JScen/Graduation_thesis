import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# ============ 日本語フォント設定（Mac の Hiragino） ============
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
font_manager.fontManager.addfont(font_path)
jp_font = font_manager.FontProperties(fname=font_path)
matplotlib.rcParams["font.family"] = jp_font.get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

# ============ データ ============
models = ["ロジスティック\n回帰", "SVM", "XGBoost", "決定木", "ランダム\nフォレスト"]
A = [0.696, 0.598, 0.578, 0.304, 0.216]   # class_weight + LOOCV
B = [0.608, 0.608, 0.255, 0.324, 0.284]   # SMOTE + 10-fold

x = np.arange(len(models))
w = 0.36

fig, ax = plt.subplots(figsize=(10, 5.5))
b1 = ax.bar(x - w/2, A, w, label="A：class_weight + LOOCV", color="#4C72B0")
b2 = ax.bar(x + w/2, B, w, label="B：SMOTE + 10-fold", color="#DD8452")

# 棒の上に数値を表示
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9)

ax.set_ylabel("感度（失踪者の検出率）", fontsize=12)
ax.set_title("2つの実験フレームワークにおける感度の比較", fontsize=13, pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 0.82)
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("sensitivity_compare.png", dpi=200, bbox_inches="tight")
print("sensitivity_compare.png を保存しました")
plt.show()
