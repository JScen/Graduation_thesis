#!/usr/bin/env python3
"""
卒論リポジトリ整理スクリプト
========================================
機能：
  1. 不要ファイルを archive/ フォルダに移動（削除しない／git履歴を保持）
  2. 主要スクリプトに番号プレフィックスを付けてリネーム（実行順が一目で分かる）
  3. まず計画を表示し、yes を入力した場合のみ実行

使い方：
  リポジトリのルート（Graduation_thesis/）で実行
    python organize_repo.py

注意：
  - .py スクリプト同士に import 依存はなく、CSV ファイル名も変更しないため、
    リネームによるリンク切れは発生しない（確認済み）
  - すべて git mv を使うので、変更は git status で確認でき、元に戻せる
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ============================================================
# ① リネーム計画：主要スクリプトに実行順の番号を付ける
#    （CSV / 画像などの成果物ファイル名は一切変更しない）
# ============================================================
RENAME = {
    "main.py":                    "01_load_data.py",
    "clean2.py":                  "02_clean_data.py",
    "LOOCV_ALL_2.py":             "03_train_models_loocv.py",
    "重要度.py":                   "04_feature_importance.py",
    "ロジスティック回帰係数.py":      "05_logreg_coefficients.py",
    "maketree.py":                "06_visualize_tree.py",
    "LOOCV_FN2.py":               "07_collect_fn.py",
    "analyze_common_FN.py":       "08_analyze_common_fn.py",
    "analyze_FN_overlap.py":      "09_analyze_fn_overlap.py",
    "analyze_FN_3groups_PPT.py":  "10_analyze_fn_3groups.py",
    "chi2_clean_chart.py":        "11_chi2_chart.py",
    # --- 追加実験（SMOTE / XGBoost） ---
    "ALL_SMOTE_10fold_2.py":      "12_smote_10fold_compare.py",
    "SMOTE_XGB_2.py":             "13_smote_xgboost.py",
    "XGB_LOOCV_2.py":             "14_xgboost_loocv.py",
    "make_compare_chart.py":      "15_make_compare_chart.py",
}


# ============================================================
# ② アーカイブ計画：archive/ 内のサブフォルダごとに分類
# ============================================================
ARCHIVE = {
    # 旧版の単一モデルスクリプト（LOOCV_ALL_2 に統合済み）
    "archive/old_scripts": [
        "Tree.py", "Logistic_regression.py", "Random_forest.py", "Svm.py",
        "TTT.py", "画树.py", "LOOCV_FN.py", "analyze_FN_for_PPT.py",
    ],
    # 早期の探索的分析（重複・カウント系）
    "archive/exploration": [
        "fandsame.py", "missfand1.py", "missfand2.py", "missfand3.py",
        "kazu.py",
    ],
    # 中間生成物の CSV（再生成可能 / 新版に置換済み）
    "archive/intermediate_csv": [
        "clean.csv", "backup_raw.csv",
        "jyuhuku.csv", "jyuhukulost.csv", "jyuhukumix.csv", "jyuhukusafe.csv",
        "miss1.csv", "miss2.DecisionTree.csv", "miss2.LogisticRegression.csv",
        "miss2.RandomForest.csv", "miss2.SVM.csv",
        "kazu.csv", "same people.csv", "tree.csv",
    ],
    # 旧版の FN ファイル（_2 なし版）
    "archive/old_fn": [
        "LOOCV_FN_Tree.csv", "LOOCV_FN_Rf.csv",
        "LOOCV_FN_Lr.csv", "LOOCV_FN_Svm.csv",
    ],
    # 旧版モデルの出力
    "archive/old_outputs": [
        "logistic_regression.csv", "random_forest.csv", "svm.csv",
        "LogisticRegression_系数.csv",
        "LogisticRegression_top_features.csv", "RandomForest_top_features.csv",
        "SVM_linear_top_features.csv", "DecisionTree_top_features.csv",
        "decision_tree", "decision_tree.pdf", "decision_tree.png",
        "decision_tree_full.txt", "decision_tree_text.txt",
        "DecisionTree_full.pdf", "DecisionTree_full.png",
    ],
    # 用途不明 / 一時ファイル
    "archive/misc": [
        "Book1.xlsx", "svmkakuritu_test.csv",
    ],
    # 初期テストコード
    "archive/test_old": [
        "test",
    ],
}


def run(cmd):
    """git コマンドを実行"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠ 失敗: {' '.join(cmd)}")
        print(f"    {result.stderr.strip()}")
        return False
    return True


def git_move(src, dst):
    """git mv を試みる。git 管理外なら通常の mv にフォールバック"""
    src_p = ROOT / src
    if not src_p.exists():
        print(f"  － スキップ（存在しない）: {src}")
        return
    dst_p = ROOT / dst
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    if run(["git", "mv", str(src_p), str(dst_p)]):
        print(f"  ✓ {src}  →  {dst}")
    else:
        # git 管理外のファイルは普通に移動
        import shutil
        shutil.move(str(src_p), str(dst_p))
        print(f"  ✓ (mv) {src}  →  {dst}")


def main():
    # --- 計画を表示 ---
    print("=" * 64)
    print("【整理計画】")
    print("=" * 64)

    print("\n■ リネーム（主要スクリプト）")
    for src, dst in RENAME.items():
        mark = " " if (ROOT / src).exists() else "✗ファイルなし"
        print(f"  {src}  →  {dst}  {mark}")

    print("\n■ アーカイブ（archive/ へ移動）")
    for folder, files in ARCHIVE.items():
        print(f"\n  [{folder}]")
        for f in files:
            mark = "" if (ROOT / f).exists() else "  ✗ファイルなし"
            print(f"    {f}{mark}")

    print("\n" + "=" * 64)
    print("※ ファイルは削除されず archive/ に移動するだけです（git mv 使用）")
    print("※ CSV・画像の成果物ファイル名は変更しません")
    print("=" * 64)

    ans = input("\nこの計画を実行しますか？ (yes/no): ").strip().lower()
    if ans != "yes":
        print("中止しました。何も変更していません。")
        return

    # --- リネーム実行 ---
    print("\n--- リネーム実行 ---")
    for src, dst in RENAME.items():
        git_move(src, dst)

    # --- アーカイブ実行 ---
    print("\n--- アーカイブ実行 ---")
    for folder, files in ARCHIVE.items():
        for f in files:
            git_move(f, f"{folder}/{Path(f).name}")

    print("\n" + "=" * 64)
    print("完了しました。")
    print("次のステップ：")
    print("  1. git status で変更を確認")
    print("  2. 問題なければ:")
    print("       git add -A")
    print('       git commit -m "リポジトリ整理：旧ファイルをarchiveへ、主要スクリプトをリネーム"')
    print("       git push")
    print("  3. もし元に戻したい場合（commit前なら）:")
    print("       git reset --hard HEAD")
    print("=" * 64)


if __name__ == "__main__":
    main()
