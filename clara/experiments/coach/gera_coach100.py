import pandas as pd
import shutil
import os

# Caminhos
csv_path = "Coach100_bench.csv"
source_dir = "/home/vinicius/Desktop/deep-grasp/experiments/coach/full_coach"
target_dir = "/home/vinicius/Desktop/deep-grasp/experiments/coach/coach100"

# Ler CSV
df = pd.read_csv(csv_path)

# Verificar se a coluna 'PDB' existe
if "PDB" not in df.columns:
    raise ValueError("A coluna 'PDB' não existe no CSV.")

# Criar a pasta de destino se não existir
os.makedirs(target_dir, exist_ok=True)

# Copiar arquivos
for pdb_id in df["PDB"].dropna().unique():
    filename = f"{pdb_id}.pdb"
    src = os.path.join(source_dir, filename)
    dst = os.path.join(target_dir, filename)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copiado: {filename}")
    else:
        print(f"Arquivo não encontrado: {filename}")
