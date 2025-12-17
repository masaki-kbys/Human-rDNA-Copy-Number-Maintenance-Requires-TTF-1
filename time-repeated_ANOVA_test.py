import pandas as pd
import pingouin as pg
import scipy.stats as stats
import io
import os

# ---------------------------------------------------------
# 1. 入出力設定
# ---------------------------------------------------------
# 'gg' : Greenhouse-Geisser
# 'hf' : Huynh-Feldt
CORRECTION_METHOD = 'gg'  # ★ここで切り替え

target_directory = r"C:/put/the/data/here"
input_dir = os.path.join(target_directory, "tra_input.csv")
df = pd.read_csv(input_dir, dtype={'name': str})

# ---------------------------------------------------------
# 2. 前処理: ユニークID作成
# ---------------------------------------------------------
# 被験者IDを一意にする (Group名 + 番号)
df['subject_id'] = df['name'].astype(str) + "_" + df['rep.'].astype(str)

print(f"Running Mixed ANOVA with {CORRECTION_METHOD.upper()} correction...")

# ---------------------------------------------------------
# 2. ANOVA実行 & イプシロン計算 (改良部分)
# ---------------------------------------------------------
aov = pg.mixed_anova(
    data=df, dv='value', within='time', between='name', subject='subject_id'
)

# 指定された方法でイプシロンを計算
epsilon = pg.epsilon(
    data=df, dv='value', within='time', subject='subject_id',
    correction=CORRECTION_METHOD
)

print(f"{CORRECTION_METHOD.upper()} Epsilon: {epsilon:.4f}")

# ---------------------------------------------------------
# 3. 補正p値の計算とカラム名設定 (改良部分)
# ---------------------------------------------------------
# 動的にカラム名を決定 (例: 'p-GG', 'epsilon_gg')
col_p = f'p-{CORRECTION_METHOD.upper()}'
col_eps = f'epsilon_{CORRECTION_METHOD}'
col_sig = f'sig_{CORRECTION_METHOD.upper()}'

def calculate_corrected_p(row, eps):
    if row['Source'] == 'name':
        return row['p-unc']
    # 自由度を補正してp値を再計算
    corrected_df1 = row['DF1'] * eps
    corrected_df2 = row['DF2'] * eps
    return stats.f.sf(row['F'], corrected_df1, corrected_df2)

# 結果の格納
aov[col_eps] = epsilon
aov[col_p] = aov.apply(lambda row: calculate_corrected_p(row, epsilon), axis=1)
aov[col_sig] = aov[col_p] < 0.05

# ---------------------------------------------------------
# 4. 保存
# ---------------------------------------------------------
# 必要な列だけ選んで保存
target_cols = ['Source', 'SS', 'DF1', 'DF2', 'MS', 'F', 'p-unc', col_eps, col_p, col_sig, 'np2']
final_result = aov[target_cols]

try:
    output_dir = os.path.join(target_directory, "ANOVA_result.csv")
    final_result.to_csv(output_dir, index=False)
    print(f"\nSuccessfully saved results")
    print(f"\n--- Result Preview ({CORRECTION_METHOD.upper()}) ---")

except Exception as e:
    print(f"Error saving CSV: {e}")