import pandas as pd
import pingouin as pg
import io
import os

# ---------------------------------------------------------
# 1. 入出力設定
# ---------------------------------------------------------
# 入力ファイルのパス (適宜書き換えてください)
# 実際のCSVを使う場合はこちらを有効にしてください
target_directory = r"C:/put/the/data/here"
input_dir = os.path.join(target_directory, "tra_input.csv")
df = pd.read_csv(input_dir, dtype={'name': str})

# ---------------------------------------------------------
# 2. 前処理: ユニークID作成
# ---------------------------------------------------------
df['subject_id'] = df['name'].astype(str) + "_" + df['rep.'].astype(str)

# ---------------------------------------------------------
# 3. Mauchlyの検定 (pg.sphericity)
# ---------------------------------------------------------
# pg.sphericity は (判定結果(bool), W値, chi2, dof, p値) の5つを返します
spher_check, W, chi2, dof, pval = pg.sphericity(
    data=df,
    dv='value',
    within='time',
    subject='subject_id'
)

# ---------------------------------------------------------
# 4. イプシロン補正係数の計算
# ---------------------------------------------------------
eps_gg = pg.epsilon(data=df, dv='value', within='time', subject='subject_id', correction='gg')
eps_hf = pg.epsilon(data=df, dv='value', within='time', subject='subject_id', correction='hf')

# ---------------------------------------------------------
# 5. 結果をDataFrameにまとめてCSV保存
# ---------------------------------------------------------
# 取得した値を辞書形式でまとめます
results = {
    'W': [W],
    'chi2': [chi2],
    'dof': [dof],
    'p-val': [pval],
    'epsilon_gg': [eps_gg],
    'epsilon_hf': [eps_hf],
    'sphericity_assumed': [spher_check] # Trueなら球面性あり、Falseならなし
}

result_df = pd.DataFrame(results)

try:
    output_dir = os.path.join(target_directory, "mauchly_result.csv")
    result_df.to_csv(output_dir, index=False)
    print(f"Successfully saved results")
    print("\n--- Result Preview ---")
    print(result_df)
except Exception as e:
    print(f"Error saving CSV: {e}")