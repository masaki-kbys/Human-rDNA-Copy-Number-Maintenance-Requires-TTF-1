import pandas as pd
import matplotlib.pyplot as plt
import io
import os


# ---------------------------------------------------------
# 1. 前処理関数 (指定されたもの)
# ---------------------------------------------------------
def preprocess_dataframe(input_df):
    """
    データフレームの前処理を行う関数
    1. 欠損値 (NaN) を含む行を削除
    2. 'name'列の '_' 以降を削除してグループ名を統合
    """
    df_tmp = input_df.copy()
    df_tmp = df_tmp.dropna()
    # 文字列型にしてからsplit
    df_tmp['name'] = df_tmp['name'].astype(str).str.split('_').str[0]
    return df_tmp


# ---------------------------------------------------------
# 2. データの準備 (動作確認用)
# ---------------------------------------------------------
# line列: "." は点のみ, "-" は隣と線をつなぐ
"""
csv_data = name,value,line
-0_1,500,.
-0_2,475,.
+1,250,-
+1_2,230,-
+2,600,-
+3,580,-
control,400,.
final,420,-
final_2,410,-
"""

FILE_DIR_in = r"C:/put/the/data/here"
FILE_DIR_out = r"C:/put/the/data/here/output"
FILE_NAME_in = 'input_g.csv'
FILE_NAME_out_fig = "Fig_2B.jpg"

file_path_in = os.path.join(FILE_DIR_in, FILE_NAME_in)
file_path_out_fig = os.path.join(FILE_DIR_out, FILE_NAME_out_fig)


raw_df = pd.read_csv(file_path_in,dtype={'name': str})

title = None
x_label = 'days'
y_label = "Hedge's G"
plt.rcParams['font.size'] = 20

# ---------------------------------------------------------
# 3. 集計と描画用データの作成
# ---------------------------------------------------------
# 前処理の実行
df = preprocess_dataframe(raw_df)

# グループの出現順序を保持するために unique() を使用
unique_groups = df['name'].unique()

plot_data = []

for group in unique_groups:
    # そのグループのデータを抽出
    sub_df = df[df['name'] == group]

    # 平均値
    mean_val = sub_df['value'].mean()

    # 標準誤差 (データ数が1以下の場合は 0 または NaN になる)
    # n=1の場合はエラーバーを出さないため 0 に設定するか、計算結果(NaN)をそのまま使う
    if len(sub_df) > 1:
        sem_val = sub_df['value'].sem()
    else:
        sem_val = 0  # n=1ならエラーバーなし(長さ0)とする

    # line列の情報 (グループ内で統一されていると仮定し、最初の行を取得)
    line_style = sub_df['line'].iloc[0]

    plot_data.append({
        'name': group,
        'mean': mean_val,
        'sem': sem_val,
        'line': line_style
    })

plot_df = pd.DataFrame(plot_data)

# ---------------------------------------------------------
# 4. グラフの描画
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

# X軸の数値を 0, 1, 2... で管理（カテゴリカルデータのプロット用）
x_indices = range(len(plot_df))
means = plot_df['mean']
sems = plot_df['sem']

# --- A. エラーバー付きプロット (全点描画) ---
# fmt='o': 線なし、点のみ。線は後で条件付きで引くため
plt.errorbar(x_indices, means, yerr=sems, fmt='o', capsize=5,
             color='black', ecolor='black', markersize=8, label='Mean ± SE')

# --- B. 条件付きで折れ線を描画 ---
# 隣り合う2点 (i, i+1) の両方が line == '-' なら線を引く
for i in range(len(plot_df) - 1):
    current_style = plot_df.loc[i, 'line']
    next_style = plot_df.loc[i + 1, 'line']

    if current_style == '-' and next_style == '-':
        # i と i+1 を結ぶ線を引く
        plt.plot([x_indices[i], x_indices[i + 1]],
                 [means[i], means[i + 1]],
                 color='black', linestyle='-')

# --- C. グラフの装飾 ---
if title:
    plt.title(title)
plt.xlabel(x_label)
plt.xticks(x_indices, plot_df['name'], fontsize = 'small')  # X軸ラベルをグループ名にする
plt.ylabel(y_label)
plt.yticks(fontsize = 'small')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 出力ディレクトリが存在しない場合は作成 (安全策)
if not os.path.exists(os.path.dirname(file_path_out_fig)):
    try:
        os.makedirs(os.path.dirname(file_path_out_fig))
    except OSError:
        pass  # ディレクトリ作成権限がない場合などは無視

plt.savefig(file_path_out_fig, dpi=300, bbox_inches='tight') # ファイル保存時はコメントアウト解除
plt.show()