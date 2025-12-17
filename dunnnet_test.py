import pandas as pd
from scipy import stats
import io
import os


# ---------------------------------------------------------
# 1. 前処理関数 (以前作成したものを再利用)
# ---------------------------------------------------------
def preprocess_dataframe(input_df):
    """
    データフレームの前処理
    1. 欠損値 (NaN) を含む行を削除
    2. 'name'列の '_' 以降を削除してグループ名を統合
    """
    df = input_df.copy()
    df = df.dropna()
    # 文字列型にしてからsplit
    df['name'] = df['name'].astype(str).str.split('_').str[0]
    return df


# ---------------------------------------------------------
# 2. データの準備
# ---------------------------------------------------------
# 実際のファイル名を指定してください
target_directory = r"C:/put/the/data/here"
input_dir = os.path.join(target_directory, "input.csv")
raw_df = pd.read_csv(input_dir, dtype={'name': str})

# ---------------------------------------------------------
# 3. データの整理
# ---------------------------------------------------------
df = preprocess_dataframe(raw_df)

# グループの出現順序を取得 (unique()は出現順を保持します)
unique_groups = df['name'].unique()

# --- Controlと実験群の特定 ---
# 最初のグループをControlとする
control_name = unique_groups[0]
treatment_names = unique_groups[1:]

print(f"Control Group: {control_name}")
print(f"Treatment Groups: {treatment_names}")

# --- 検定用データの抽出 ---
# Control群のデータ配列
control_data = df[df['name'] == control_name]['value'].values

# 実験群のデータ配列をリストに格納
treatment_datas = []
for t_name in treatment_names:
    t_data = df[df['name'] == t_name]['value'].values
    treatment_datas.append(t_data)

# ---------------------------------------------------------
# 4. Dunnett検定の実行
# ---------------------------------------------------------
# scipy.stats.dunnett (SciPy 1.11.0以降で利用可能)
# 引数: (*samples, control) -> samplesは実験群のリストを展開して渡す
res = stats.dunnett(*treatment_datas, control=control_data)

# ---------------------------------------------------------
# 5. 結果の出力
# ---------------------------------------------------------
results = []

# res.pvalue は実験群の順番通りにp値が入っています
for i, t_name in enumerate(treatment_names):
    p_val = res.pvalue[i]

    results.append({
        'group_name_1': control_name,  # Control
        'group_name_2': t_name,  # Treatment
        'p-value': p_val
    })

output_df = pd.DataFrame(results)

print("\n--- Dunnett's Test Result ---")
print(output_df)

# CSV保存
output_dir = os.path.join(target_directory, "dunnett_result.csv")
output_df.to_csv(output_dir, index=False)