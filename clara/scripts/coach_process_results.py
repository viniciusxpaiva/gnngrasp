import os
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from imblearn.metrics import geometric_mean_score

test_name = "teste"
test_range = "1-15"


def load_coach_ground_truth(labels_path):
    """
    Load COACH ground truth from labels.txt
    Returns a dict: (pdb_id, chain_id) -> set(residue_numbers)
    """
    gt = {}
    with open(labels_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            prot_info, residues = parts
            pdb_chain = prot_info[:5]  # e.g., 1a8tA
            pdb_id = pdb_chain[:4].lower()
            chain_id = pdb_chain[4].upper()
            residue_numbers = set(
                int(r) for r in residues.split(",") if r.strip().isdigit()
            )

            key = (pdb_id, chain_id)
            if key not in gt:
                gt[key] = set()
            gt[key].update(residue_numbers)  # append all from current line

    return gt


def evaluate_coach_predictions(predictions_dir, labels_path):
    """
    Evaluate all predictions against COACH ground truth labels.
    """
    ground_truth = load_coach_ground_truth(labels_path)

    all_true = []
    all_pred = []

    for folder in os.listdir(predictions_dir):
        folder_path = os.path.join(predictions_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if not file.endswith("_prediction.csv"):
                continue

            path = os.path.join(folder_path, file)
            filename = os.path.splitext(file)[0].replace("_prediction", "").upper()
            pdb_id = filename[:4].lower()
            chain_id = filename[4]

            # print(f"[✓] Evaluating {pdb_id}{chain_id}...")

            if (pdb_id, chain_id) not in ground_truth:
                print(f"[!] No ground truth for {pdb_id}{chain_id}. Skipping.")
                continue

            try:
                bs_residues = ground_truth[(pdb_id, chain_id)]
                df_pred = pd.read_csv(path)

                true_labels = []
                pred_labels = []

                for row in df_pred.itertuples():
                    # with open("tmp.log", "a") as f:
                    #    f.write(f"{filename}_{row.residue_id}\n")
                    parts = row.residue_id.split("_")  # e.g., PRO_615_A
                    res_num = int(parts[1])
                    res_chain = parts[2].upper()

                    if res_chain != chain_id:
                        continue  # skip other chains if present

                    label = 1 if res_num in bs_residues else 0
                    true_labels.append(label)
                    pred_labels.append(int(row.predicted_label))

                all_true.extend(true_labels)
                all_pred.extend(pred_labels)

            except Exception as e:
                print(f"[X] Failed on {pdb_id}{chain_id}: {e}")

    # --- Metrics ---
    if not all_true:
        print("[!] No valid predictions to evaluate.")
        return

    mcc = matthews_corrcoef(all_true, all_pred)
    precision = precision_score(all_true, all_pred)
    recall = recall_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)
    auc = roc_auc_score(all_true, all_pred)
    # gmean = geometric_mean_score(all_true, all_pred)

    return {
        "MCC": mcc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc,
        # "GMean": gmean,
    }


# === Example usage ===
if __name__ == "__main__":
    labels_path = "../experiments/coach/labels.txt"  # path to COACH labels
    begin = int(test_range.split("-")[0])
    end = int(test_range.split("-")[1])
    print(f"[→] Evaluating test {test_name} from {begin} to {end}...")
    rows = []  # to collect per-test metrics
    for i in range(begin, end + 1):
        test_id = f"{test_name}{i}"
        predictions_dir = f"../{test_id}_output"  # path to output predictions
        metrics = evaluate_coach_predictions(predictions_dir, labels_path)
        if metrics is not None:
            metrics["Test"] = test_id
            rows.append(metrics)

    if not rows:
        print("[!] No tests produced valid metrics.")
    else:
        # Build a DataFrame with all tests and sort by F1 descending
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values(by="F1", ascending=False).reset_index(drop=True)

        # Print full ranked table with more spacing between columns
        print("\n=== All tests ranked by F1 (desc) ===")
        print(
            df_sorted.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
                col_space=10,  # minimum column width
            )
        )
        # Compute averages across all tests
        avg_metrics = df_sorted.drop(columns=["Test"]).mean(numeric_only=True)

        print("\n=== Average metrics across all tests ===")
        for metric, value in avg_metrics.items():
            print(f"{metric:<10}: {value:.3f}")
