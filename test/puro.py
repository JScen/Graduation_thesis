import matplotlib.pyplot as plt
import networkx as nx

# 研究計画のフローチャートノード定義
labels = {
    0: "研究目的\n失踪傾向の明確化",
    1: "定量的分析\n原因の特定",
    2: "属性分析\n(年齢, 職種, 在日年数, 語学学校)",
    3: "モデル構築\n機械学習 (LR/RF)",
    4: "早期把握\n失踪リスク予測"
}

# グラフ作成
G = nx.DiGraph()
for i in labels:
    G.add_node(i)
for i in range(len(labels)-1):
    G.add_edge(i, i+1)

# レイアウト設定
pos = {i: (i*2, 0) for i in labels}

# 描画
plt.figure(figsize=(12, 2))
nx.draw(G, pos, with_labels=False, arrows=True, node_size=3500, node_color='lightblue')

# ノードラベルの描画
for i, label in labels.items():
    x, y = pos[i]
    plt.text(x, y, label, ha='center', va='center', fontsize=10)

plt.axis('off')
plt.tight_layout()
plt.show()