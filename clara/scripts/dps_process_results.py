import os
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score

AA1_to_AA3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


def evaluate_all_predictions(fasta_path, predictions_dir):
    """
    Evaluate all predictions in the given directory using ground truth from the FASTA file.

    Args:
        fasta_path (str): Path to the FASTA file.
        predictions_dir (str): Path to the directory containing prediction CSVs (subfolders).
    """
    # --- Step 1: Parse FASTA file and extract ground truth labels ---
    ground_truth = {}  # maps (pdb_id, chain_id) -> (residue_ids, labels)

    with open(fasta_path) as f:
        lines = f.read().splitlines()

    for i in range(0, len(lines), 3):  # every 3 lines: header, seq, labels
        header = lines[i].strip()[1:].upper()  # remove ">"
        seq = lines[i + 1].strip()
        labels = lines[i + 2].strip()

        if len(seq) != len(labels):
            raise ValueError(f"Label length mismatch for {header}")

        pdb_id = header[:4]
        chain_id = header[4:]

        residue_ids = [
            f"{AA1_to_AA3.get(res, 'UNK')}_{idx+1}_{chain_id}"
            for idx, res in enumerate(seq)
        ]
        ground_truth[(pdb_id, chain_id)] = (residue_ids, [int(x) for x in labels])

    # --- Step 2: Loop over prediction files ---
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
            filename = os.path.splitext(file)[0]  # e.g., 4yocA_prediction
            pdb_chain = filename.replace("_prediction", "").upper()
            pdb_id = pdb_chain[:4]
            chain_id = pdb_chain[4:]

            print(f"[✓] Evaluating {pdb_id}{chain_id}...")

            # Skip if ground truth is missing
            if (pdb_id, chain_id) not in ground_truth:
                print(
                    f"[!] Ground truth for {pdb_id}{chain_id} not found in FASTA. Skipping."
                )
                continue

            try:
                residue_ids_gt, labels_gt = ground_truth[(pdb_id, chain_id)]
                df_pred = pd.read_csv(path)

                if len(df_pred) != len(labels_gt):
                    print(f"[!] Length mismatch in {pdb_id}{chain_id}, skipping.")
                    print(len(df_pred), len(labels_gt))
                    continue

                # --- Step 4: Validate alignment (ignore residue number) ---
                # Compare only residue name (3-letter) and chain ID
                for i, pred_row in enumerate(df_pred.itertuples()):
                    pred_parts = pred_row.residue_id.split("_")
                    pred_resname, pred_chain = pred_parts[0], pred_parts[-1]

                    gt_parts = residue_ids_gt[i].split("_")
                    gt_resname, gt_chain = gt_parts[0], gt_parts[-1]

                    if pred_resname != gt_resname or pred_chain != gt_chain:
                        raise ValueError(
                            f"Residue mismatch at position {i}: {pred_row.residue_id} vs {residue_ids_gt[i]}"
                        )

                labels_pred = df_pred["predicted_label"].astype(int).tolist()

                all_true.extend(labels_gt)
                all_pred.extend(labels_pred)

            except Exception as e:
                print(f"[X] Failed on {pdb_id}{chain_id}: {e}")

    # --- Step 3: Calculate overall metrics ---
    if not all_true:
        print("No valid predictions found.")
        return

    mcc = matthews_corrcoef(all_true, all_pred)
    precision = precision_score(all_true, all_pred)
    recall = recall_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)

    print("\n=== Global Evaluation Metrics ===")
    print(f"MCC      : {mcc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


# === Example usage ===
if __name__ == "__main__":
    fasta_path = "../experiments/deepprosite/tmp_test.fa"
    predictions_dir = "../output/"
    evaluate_all_predictions(fasta_path, predictions_dir)
