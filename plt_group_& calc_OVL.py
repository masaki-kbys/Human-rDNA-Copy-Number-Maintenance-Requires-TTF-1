import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import ast
import os

# ---------------------------------------------------------
# 1. 準備：読み込み&書き出し用のファイル名を指定
# ---------------------------------------------------------

#csvの中身はこんな感じに
"""csv_content = name, average, stdev, line_style
m0, 308.26, 28.0429,
p0, 249.7, 35.0807, --
p2, 336.572, 86.4424, -.
p4, 257.798, 136.17, :
p6, 295.314, 37.5794, "(0, (3, 1, 1, 1))"
"""
FILE_DIR_in = r"C:/put/the/data/here"  # データのインプットもとのパスを入力
FILE_DIR_out = r"C:/put/the/data/here/output"  # データのアウトプット先のパスを入力

FILE_NAME_in = 'inputd.csv'  # inputファイル名を入力
FILE_NAME_out_csv = "distribution_results.csv"  # outputファイル名を入力
FILE_NAME_out_fig = "distribution_graph.png"  # outputファイル名を入力

file_path_in = os.path.join(FILE_DIR_in, FILE_NAME_in)
file_path_out_csv = os.path.join(FILE_DIR_out, FILE_NAME_out_csv)
file_path_out_fig = os.path.join(FILE_DIR_out, FILE_NAME_out_fig)

input_csv = file_path_in
output_csv = file_path_out_csv  # 結果出力用
output_img = file_path_out_fig  # 画像保存用

x_label ='CN' #グラフのx軸のタイトル
y_label = 'Relative Intensity (Peak=1)' #グラフのyラベル

print(f"[{input_csv}] を読み込みます...")

# ---------------------------------------------------------
# 2. データ処理と計算準備
# ---------------------------------------------------------
df = pd.read_csv(input_csv)
df.columns = [c.strip() for c in df.columns]  # 空白除去

# 新しい列「overlap_coefficient」を作成（初期値は0.0）
df['overlap_coefficient'] = 0.0


# 線種解析関数
def parse_linestyle(style_str):
    if pd.isna(style_str) or str(style_str).strip() == "":
        return '-'
    s = str(style_str).strip()
    if s.startswith('(') or s.startswith('"('):
        try:
            return ast.literal_eval(s.replace('"', ''))
        except:
            return '-'
    return s


# 描画範囲の自動決定
x_mins = df['average'] - 4 * df['stdev']
x_maxs = df['average'] + 4 * df['stdev']
x_range = np.linspace(min(x_mins), max(x_maxs), 2000)
dx = x_range[1] - x_range[0]

# 基準データ（1行目）の取得と計算
ref_idx = df.index[0]
ref_row = df.iloc[0]
y_ref_raw = norm.pdf(x_range, ref_row['average'], ref_row['stdev'])
y_ref_norm = y_ref_raw / np.max(y_ref_raw)

# 基準データのオーバーラップ係数は1.0 (100%) とする
df.at[ref_idx, 'overlap_coefficient'] = 1.0

# ---------------------------------------------------------
# 3. グラフ描画ループ
# ---------------------------------------------------------
plt.figure(figsize=(12, 8))

# --- 基準（m0）の描画 ---
plt.plot(x_range, y_ref_norm,
         color='black', linewidth=3, linestyle='-',
         label=f"{ref_row['name']} (Reference)", zorder=10)

# --- 比較対象の計算と描画 ---
for index, row in df.iloc[1:].iterrows():
    # 計算
    y_target_raw = norm.pdf(x_range, row['average'], row['stdev'])

    # オーバーラップ計算 (m0 との共通面積)
    overlap = np.sum(np.minimum(y_ref_raw, y_target_raw)) * dx

    # ★結果をDataFrameに保存
    df.at[index, 'overlap_coefficient'] = overlap

    # 描画用正規化
    y_target_norm = y_target_raw / np.max(y_target_raw)
    ls = parse_linestyle(row['line_style'])

    # プロット
    plt.plot(x_range, y_target_norm,
             color='0.4',
             linewidth=2,
             linestyle=ls,
             label=f"{row['name']} (Overlap: {overlap:.1%})")

    # 塗りつぶし
    plt.fill_between(x_range, np.minimum(y_ref_norm, y_target_norm),
                     color='gray', alpha=0.1)

# ---------------------------------------------------------
# 4. グラフの装飾と保存
# ---------------------------------------------------------
plt.title(f"Distribution Overlap with {ref_row['name']}", fontsize=15)
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.legend(title="Group (Line Style / Overlap %)", fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(min(x_mins), max(x_maxs))

# ★画像の保存
plt.tight_layout()
plt.savefig(output_img, dpi=300, bbox_inches='tight')  # 300dpiで高画質保存
print(f"グラフを保存しました: {output_img}")

plt.show()

# ---------------------------------------------------------
# 5. 【追加機能】個別グラフの作成と保存
# ---------------------------------------------------------
print("\n--- 個別グラフの生成を開始します ---")

# 比較対象（2行目以降）について再度ループ
for index, row in df.iloc[1:].iterrows():
    # 新しい描画領域を作成（毎回リセット）
    plt.figure(figsize=(10, 6))

    # 1. 基準（m0）を描画
    plt.plot(x_range, y_ref_norm,
             color='black', linewidth=3, linestyle='-',
             label=f"{ref_row['name']} (Reference)", zorder=10)

    # 2. 比較対象のデータを再計算・描画
    y_target_raw = norm.pdf(x_range, row['average'], row['stdev'])
    y_target_norm = y_target_raw / np.max(y_target_raw)
    ls = parse_linestyle(row['line_style'])

    # 既に計算済みのオーバーラップ率を取得
    overlap = df.at[index, 'overlap_coefficient']

    plt.plot(x_range, y_target_norm,
             color='0.4', linewidth=2, linestyle=ls,
             label=f"{row['name']} (Overlap: {overlap:.1%})")

    # 3. 塗りつぶし
    plt.fill_between(x_range, np.minimum(y_ref_norm, y_target_norm),
                     color='gray', alpha=0.1)

    # 4. 装飾（全体図とスケールを合わせる）
    plt.title(f"Individual Comparison: {ref_row['name']} vs {row['name']}", fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(min(x_mins), max(x_maxs))  # X軸の範囲を全体図と統一して比較しやすくする

    # 5. 個別ファイルとして保存
    # ファイル名例: compare_m0_vs_p0.png
    filename = f"compare_{ref_row['name']}_vs_{row['name']}.png"
    file_path_out_ind_fig = os.path.join(FILE_DIR_out, filename)
    plt.savefig(file_path_out_ind_fig, dpi=300, bbox_inches='tight')
    plt.close()  # メモリ開放のため描画を閉じる

    print(f"個別のグラフを保存しました: {filename}")

# ---------------------------------------------------------
# 6. CSVデータの保存
# ---------------------------------------------------------
# 結果を見やすく整形（小数点4桁など）したければここで行うが、今回は生の値を出力
df.to_csv(output_csv, index=False)
print(f"結果CSVを保存しました: {output_csv}")

# 確認のため先頭を表示
print("\n--- Output Data Preview ---")
print(df[['name', 'average', 'stdev', 'overlap_coefficient']])