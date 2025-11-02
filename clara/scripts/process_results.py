import os
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


residues_with_labels_file = "../experiments/coach/coach_residues_with_labels.csv"  # Arquivo com os labels
test_code = "t7"


def get_grasp_results(output_dir, residues_with_labels_file, result_file):
    # 1. Ler o arquivo de resíduos com labels
    residues_df = pd.read_csv(residues_with_labels_file)

    # 2. Criar uma nova coluna para a predição e inicializar com 0
    residues_df["predicted_label"] = 0
    conta = 0
    entrou = 0
    # 3. Percorrer os arquivos de saída do GRaSP
    cont = 0
    total = len(os.listdir(output_dir))
    for res_folder in os.listdir(output_dir):
        # print(f"Processando: {res_folder}")
        if cont % 50 == 0:
            print(f"Processing results {cont+1}/{total}")
        cont += 1

        pdb_id = res_folder.split("_")[0]
        chain = res_folder.split("_")[1]

        # Caminho para o arquivo CSV do GRaSP
        csv_path = os.path.join(
            output_dir, res_folder, f"{pdb_id}{chain}_prediction.csv"
        )

        # Verificar se o arquivo existe
        if os.path.isfile(csv_path):
            # Ler os resultados do GRaSP
            entrou += 1
            grasp_df = pd.read_csv(csv_path)
            grasp_predicted = grasp_df[grasp_df["predicted_label"] == 1]

            if grasp_predicted.empty:
                conta += 1

            # Atualizar a coluna 'prediction' no DataFrame principal
            for _, row in grasp_predicted.iterrows():
                res_str = row["residue_id"].strip()  # Ex: GLN_8_C

                try:
                    # Separar partes
                    parts = res_str.split("_")
                    resname = parts[0]  # Ex: GLN
                    resnum = parts[1]  # Ex: 8
                    chain = parts[2]  # Ex: C

                    # Agora montar no formato que bate com o residues_with_labels_file
                    formatted_residue_id = f"{pdb_id}{chain}_{chain}_{resname}_{resnum}"

                    # Atualizar a predição
                    residues_df.loc[
                        residues_df["residue_id"] == formatted_residue_id,
                        "predicted_label",
                    ] = 1

                except Exception as e:
                    print(f"[WARNING] Failed to parse residue_id: {res_str} ({str(e)})")

        else:
            print(csv_path)

    # 4. Salvar o DataFrame atualizado em um arquivo CSV
    residues_df.to_csv(result_file, index=False)

    print(f"Resultados atualizados com predições do GRaSP salvos em '{result_file}'.")
    print("Total de vazios:", conta)
    print("entrou:", entrou)


def calculate_metrics(result_file):
    # Ler o DataFrame consolidado com a coluna 'label' e 'prediction'
    df = pd.read_csv(result_file)

    # Extrair rótulos verdadeiros e predições
    true_labels = df["label"]  # Rótulos verdadeiros
    predictions = df["predicted_label"]  # Predições binárias

    # Calcular métricas
    mcc = matthews_corrcoef(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    # Verificar se AUC pode ser calculado (se não houver apenas uma classe em `true_labels`)
    if len(set(true_labels)) > 1:
        auc = roc_auc_score(true_labels, predictions)
    else:
        auc = "N/A (AUC requires at least two classes)"

    # Exibir as métricas
    print("MCC:", round(mcc, 2))
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))
    print("F1-Score:", round(f1, 2))
    print(
        "AUC:", round(auc, 2) if isinstance(auc, (int, float)) else auc
    )  # Evita erro se AUC for "N/A"


output_dir = "../" + test_code + "_output"  # Diretório dos resultados do GRaSP
result_file = "../results/" + test_code + "_results.csv"  # Arquivo final a ser gerado

get_grasp_results(output_dir, residues_with_labels_file, result_file)
calculate_metrics(result_file)
print("----------------------------------")
