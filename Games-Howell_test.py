import pandas as pd
import pingouin as pg
import io
import os

# ---------------------------------------------------------
# 1. データの準備
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

# ---------------------------------------------------------
# 2. Games-Howell検定の実行
# ---------------------------------------------------------
# dv: dependent variable (従属変数=数値), between: グループ列
gh_results = pg.pairwise_gameshowell(data=df_preped, dv='value', between='name')

# ---------------------------------------------------------
# 3. 出力
# ---------------------------------------------------------
# pingouinの出力は 'A', 'B', 'pval' 等の列名なので、指定の形式に変更
output_df = gh_results

# p値を丸めるなどの処理が必要ならここで行う
# output_df['p-value'] = output_df['p-value'].round(5)

print("--- Games-Howell検定結果 ---")
print(output_df)

# CSVとして保存
output_dir = os.path.join(target_directory, 'Games-Howell_result.csv')
output_df.to_csv(output_dir, index=False)