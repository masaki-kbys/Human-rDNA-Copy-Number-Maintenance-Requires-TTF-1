import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------
# 1. CSVファイルの読み込み
# ---------------------------------------------------------
def prep_dataframe(input_df):
    """
    データフレームの前処理を行う関数
    1. 欠損値 (NaN) を含む行を削除
    2. 'name'列の '_' 以降を削除してグループ名を統合 (例: '-0_1' -> '-0')
    """
    # 元のデータを変更しないようにコピーを作成
    df_tmp = input_df.copy()

    # 1. 欠損値の削除
    df_tmp = df_tmp.dropna()

    # 2. name列の文字列操作
    # '_' で分割し、その最初の要素(0番目)を取得することで '_' 以降をカットします
    # 文字列型であることを保証するために astype(str) を挟んでいます
    df_tmp['name'] = df_tmp['name'].astype(str).str.split('_').str[0]

    return df_tmp

# 実際のファイル名を指定してください
target_directory = r"C:/put/the/data/here"
input_dir = os.path.join(target_directory, "input.csv")
df_org = pd.read_csv(input_dir, dtype={'name': str})
df_preped = prep_dataframe(df_org)

"""
# ※ここでは動作確認用にダミーデータを生成します
import numpy as np

np.random.seed(42)
data = {
    'name': np.repeat(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'], 24),
    'value': np.concatenate([
        np.random.normal(10, 2, 24),  # G1: 正規分布
        np.random.normal(12, 3, 24),  # G2: 正規分布
        np.random.exponential(5, 24),  # G3: 指数分布（非正規）
        np.random.normal(10, 2, 24),  # G4
        np.random.normal(11, 2, 24),  # G5
        np.random.normal(9, 1, 24)  # G6
    ])
}
df = pd.DataFrame(data)
"""

# ---------------------------------------------------------
# 2. 検定と可視化
# ---------------------------------------------------------
results = []

# グラフのスタイル設定（お好みで）
sns.set(style="whitegrid")

for group_name, group_data in df_preped.groupby('name'):
    values = group_data['value']

    # Shapiro-Wilk検定
    stat, p_value = stats.shapiro(values)
    results.append({'group_name': group_name, 'p-value': p_value})

    # 判定コメント（グラフタイトル用）
    judge = "Normal" if p_value >= 0.05 else "Non-Normal"
    color = "blue" if p_value >= 0.05 else "red"

    # --- 描画 (1行2列) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.suptitle(f'Group: {group_name} (N={len(values)}) | p={p_value:.4f} [{judge}]',
                 fontsize=14, color=color, fontweight='bold')

    # ヒストグラム
    sns.histplot(values, kde=True, ax=axes[0], color='skyblue', bins=8)
    axes[0].set_title('Histogram')

    # Q-Qプロット
    stats.probplot(values, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_markerfacecolor('blue')
    axes[1].get_lines()[0].set_markersize(6.0)
    axes[1].set_title('Q-Q Plot')

    plt.tight_layout()

    plt.show()

# ---------------------------------------------------------
# 3. 結果の出力
# ---------------------------------------------------------
result_df = pd.DataFrame(results)

# p値で判定列を追加（見やすくするため）
result_df['Significance'] = result_df['p-value'].apply(
    lambda x: 'Non-Normal (<0.05)' if x < 0.05 else 'Normal (>=0.05)')

print("\n=== 検定結果一覧 ===")
print(result_df)

# 必要であればCSV保存
output_dir = os.path.join(target_directory, 'shapiro_result.csv')
result_df.to_csv(output_dir, index=False)