import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
import os

# ---------------------------------------------------------
# 0. 描画のON/OFF設定 (Boolean配列)
# ---------------------------------------------------------
# [Violin, Swarm, Mean_Point(バツ印), Mean_Line(折れ線)]
# True: 描画する, False: 描画しない
drawing = [True, True, True, False]
# 例: [True, False, True, True] ならSwarm以外を描画

# ---------------------------------------------------------
# 1. 準備：読み込み&書き出し用のファイル名設定
# ---------------------------------------------------------

# 動作確認用ダミーデータ
csv_content = """name,value
-0, 1.2
-0, 1.5
-0, 1.8
-0, 0.9
+0_1, 2.8
+0_2, 3.1
+0_15, 2.2
+0_3, 2.4
Control, 1.5
Control_A, 1.6
Control_B, 1.4
"""

FILE_DIR_in = r"C:/put/the/data/here"
FILE_DIR_out = r"C:/put/the/data/here/output"
FILE_NAME_in = 'input.csv'
FILE_NAME_out_fig = "output.jpg"

file_path_in = os.path.join(FILE_DIR_in, FILE_NAME_in)
file_path_out_fig = os.path.join(FILE_DIR_out, FILE_NAME_out_fig)

# ★動作確認のため、ここでは文字列から読み込みます。
# 実際のファイルを使う場合は下行を有効にし、io.StringIOの行をコメントアウトしてください
df = pd.read_csv(file_path_in)
# df = pd.read_csv(io.StringIO(csv_content))

title = None
x_label = 'days'
y_label = 'rDNA CN'
plt.rcParams['font.size'] = 30

# ---------------------------------------------------------
# 2. データの前処理
# ---------------------------------------------------------
# "name"列の文字列処理
df['name'] = df['name'].astype(str).apply(lambda x: x.split('_')[0])

# ---------------------------------------------------------
# 3. グラフの描画設定
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# --- A. Violin Plot (drawing[0]) ---
if drawing[0]:
    ax = sns.violinplot(
        data=df,
        x='name',
        y='value',
        inner=None,
        color='white',
        linewidth=2,
        cut=0,
        zorder=1  # 最背面に配置
    )

    # Violinの線種変更処理
    line_styles = ['-', '-', (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1, 1, 1)), ":"]

    # 描画されているViolinのオブジェクトのみを取得して変更
    if ax.collections:
        violin_parts = ax.collections
        for i, part in enumerate(violin_parts):
            style = line_styles[i % len(line_styles)]
            part.set_edgecolor('black')
            part.set_linestyle(style)
            part.set_alpha(1.0)

# --- B. Swarm Plot (drawing[1]) ---
if drawing[1]:
    sns.swarmplot(
        data=df,
        x='name',
        y='value',
        color="0.5",
        alpha=0.7,
        size=8,
        zorder=2  # Violinより手前
    )

# --- C. Mean Line (drawing[3]) ---
# ※ 平均値を計算して線だけ引く (Mean Pointより奥、Swarmより手前くらいが見やすい)
if drawing[3]:
    sns.pointplot(
        data=df,
        x='name',
        y='value',
        estimator=np.mean,
        errorbar=None,
        markers="",  # マーカーなし
        linestyles='-',  # 実線
        color='black',
        scale=0.5,  # マーカー非表示の補助設定
        zorder=10  # 点の下に来るように設定(Pointが11なのでそれ以下)
    )

# --- D. Mean Point (drawing[2]) ---
# ※ 平均値をバツ印で表示
if drawing[2]:
    sns.pointplot(
        data=df,
        x='name',
        y='value',
        estimator=np.mean,
        errorbar=None,
        markers='x',
        color='black',
        linestyles='none',  # 線は引かない(Line側で制御するため)
        markersize=12,
        markeredgewidth=2,
        zorder=11  # 最前面
    )

# ---------------------------------------------------------
# 4. 保存と表示
# ---------------------------------------------------------
if title:
    plt.title(title)
plt.xlabel(x_label)
plt.xticks(fontsize = 'small')
plt.ylabel(y_label)
plt.yticks(fontsize = 'small')
plt.tight_layout()

# 出力ディレクトリが存在しない場合は作成 (安全策)
if not os.path.exists(os.path.dirname(file_path_out_fig)):
    try:
        os.makedirs(os.path.dirname(file_path_out_fig))
    except OSError:
        pass  # ディレクトリ作成権限がない場合などは無視

plt.savefig(file_path_out_fig, dpi=300, bbox_inches='tight') # ファイル保存時はコメントアウト解除
plt.show()