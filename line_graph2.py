import pandas as pd
import matplotlib.pyplot as plt
import io
import os


# ---------------------------------------------------------
# 1. 前処理関数 (変更なし)
# ---------------------------------------------------------
def preprocess_dataframe(input_df):
    """
    データフレームの前処理を行う関数
    1. 欠損値 (NaN) を含む行を削除
    2. 'name'列の '_' 以降を削除してグループ(系列)名を統合
    ※ name列は文字列として処理し、-0と+0などを区別する
    """
    df = input_df.copy()
    df = df.dropna()

    # name列を文字列にしてから統合
    df['name'] = df['name'].astype(str).str.split('_').str[0]

    return df


# ---------------------------------------------------------
# 2. データの準備 (動作確認用・変更なし)
# ---------------------------------------------------------
FILE_DIR_in = r"C:/put/the/data/here"
FILE_DIR_out = r"C:/put/the/data/here/output"
FILE_NAME_in = 'input_sr.csv'
FILE_NAME_out_fig = "Fig_5B.jpg"

file_path_in = os.path.join(FILE_DIR_in, FILE_NAME_in)
file_path_out_fig = os.path.join(FILE_DIR_out, FILE_NAME_out_fig)

raw_df = pd.read_csv(file_path_in,dtype={'name': str})

title = None
# y_label =" cell number [log $10^{{{5}}}$ cells/well]"
y_label = "cell survival rate [%]"
# y_label = "rDNA CN"
x_label = "days"
plt.rcParams['font.size'] = 20


# ---------------------------------------------------------
# 3. 集計処理 (変更なし)
# ---------------------------------------------------------
df = preprocess_dataframe(raw_df)
agg_df = df.groupby(['name', 'X-value']).agg({
    'Y-value': ['mean', 'sem', 'count'],
    'line': 'first'
}).reset_index()
agg_df.columns = ['name', 'X-value', 'mean', 'sem', 'count', 'line']
agg_df = agg_df.sort_values(by=['name', 'X-value'])

# ---------------------------------------------------------
# 4. グラフの描画 (★ここを変更★)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

unique_series = agg_df['name'].unique()

# --- 線スタイルとマーカーの定義 ---
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', 'v', '*', 'p']

# --- ★ずらし(Jitter)の設定 ---
jitter_step = 0.1  # 各系列間をどれくらい空けるか (調整してください)
n_series = len(unique_series)
# 全体のずらし幅の中央が元のX位置になるように開始位置を計算
start_offset = -((n_series - 1) * jitter_step) / 2

for i, series_name in enumerate(unique_series):
    sub_df = agg_df[agg_df['name'] == series_name].reset_index(drop=True)

    # 元のX値
    xs_original = sub_df['X-value']

    # ★ずらしたX値を計算
    # 現在の系列のオフセット量
    current_offset = start_offset + (i * jitter_step)
    # 元のX値にオフセットを加算 (pandas Seriesどうしの足し算)
    xs_jittered = xs_original + current_offset

    means = sub_df['mean']
    sems = sub_df['sem']
    sems_plot = sems.where(sub_df['count'] > 1, 0)

    line_style = line_styles[i % len(line_styles)]
    marker_style = markers[i % len(markers)]
    plot_color = 'black'

    # --- A. 点とエラーバーのプロット ---
    # ★ xs_original ではなく xs_jittered を使用
    plt.errorbar(xs_jittered, means, yerr=sems_plot, fmt=marker_style, capsize=5,
                 label=series_name, color=plot_color, markersize=8)

    # --- B. 条件付きで折れ線を描画 ---
    for j in range(len(sub_df) - 1):
        # ★ここでもずらしたX座標を使用 (ilocでアクセス)
        current_x = xs_jittered.iloc[j]
        next_x = xs_jittered.iloc[j + 1]

        current_mean = sub_df.loc[j, 'mean']
        next_mean = sub_df.loc[j + 1, 'mean']
        current_line_req = sub_df.loc[j, 'line']
        next_line_req = sub_df.loc[j + 1, 'line']

        if current_line_req == '-' and next_line_req == '-':
            plt.plot([current_x, next_x], [current_mean, next_mean],
                     color=plot_color, linestyle=line_style)

# ---------------------------------------------------------
# 5. グラフの装飾と保存
# ---------------------------------------------------------
if title:
    plt.title(title)
plt.xlabel(x_label)
plt.xticks(fontsize = 'small')  # X軸ラベルをグループ名にする
plt.ylabel(y_label)
plt.yticks(fontsize = 'small')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.ylim(0, 100)
plt.legend(title='Series Name')
plt.grid(True, linestyle='--', alpha=0.6)

output_filename = 'plot.png'
plt.savefig(file_path_out_fig, dpi=300, bbox_inches='tight')
print(f"Graph saved")

plt.show()