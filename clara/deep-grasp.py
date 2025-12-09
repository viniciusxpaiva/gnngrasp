import os
import time
import copy
import numpy as np
import argparse

# from app.templates.distance_edge_generator import EdgeDistanceGenerator
# from app.embeddings.esm_node_embeddings_generator import ESMNodeEmbeddingsGenerator
from app.pipeline import Pipeline
#from app.utils.experiments_utils import run_planetoid_grid_experiments

# from app.ego_graphs.templates.ego_templates_extractor import EgoGraphTemplateExtractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepBENDER pipeline")
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Output directory for predictions",
    )
    args = parser.parse_args()

    print("Starting Deep-GRaSP")

    PREDICTION_PARAMS = {
        "top_n_templates": 30,
        "sub_model": False,
        "num_gnn_layers": 2,
        "use_embd_projection": False,
        "use_edge_attr": False,
        "baseline": False,
        "hier": True,
        "smote": False,
        "node_import": False,
        "bat": False,
        "epochs": 100,
        "embd_type": "ESM",
        "layer_type": "GAT",
        "gnn_params": {
            "hidden_dim": 128,
            "lr": 0.001,
            "dropout": 0.3,
            "weight_decay": 1e-4,
            "num_heads": 6,
            "projected_embd_dim": 128,
        },
        "cnn_params": {
            "cnn_channels": 32,
            "cnn_kernel_size": 5,
            "cnn_dropout": 0.2,
        },
        "mlp_params": {
            "mlp_hidden": 32,
            "mlp_dropout": 0.2,
        },
        "prediction_threshold": 0.5,
    }

    # Configuration dictionary for hierarchical subgraph classification
    HIER_PARAMS = {
        "subg_gen_method": "color",  # asa | color | anchor
        "subg_neighbor_layers": 3,
        "use_subgraph_classifier": True,
        "use_subgraph_filtering": True,
        "multiclass_decision_rule": "logit_margin",  # "argmax" | "margin" | "logit_margin"
        # "all_or_pos_subg_node_training": "all",  # all | pos
        "top_n_templates": 30,
        "asa_exposure_percent": 60,
        "subgraph_classifier": {
            "gnn_type": "GAT",  # GCN | GAT | GIN | SAGE | PNA
            "norm_type": "batch",  #
            "num_layers": 2,
            "pool_type": "add",
            "hidden_dim": 128,
            "output_dim": 1,
            "num_heads": 4,
            "dropout": 0.2,
            "lr": 1e-4,
            "epochs": 150,
            "weight_decay": 1e-5,
            "prediction_threshold": 0.5,
            "metric_mode": "f_beta",
            "beta": 1,
        },
        "node_classifier": {
            "gnn_type": "GAT",  # GCN | GAT | GIN | SAGE
            "norm_type": "batch",
            "num_layers": 1,
            "hidden_dim": 128,
            "num_heads": 4,
            "dropout": 0.2,
            "lr": 1e-4,
            "epochs": 60,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "fusion_mode": "none",  # concat | film | none
            "context_norm": "layernorm",  # 'none' | 'layernorm' | 'batch_zscore' | 'l2'
            "context_anneal": "none",  # 'none'|'linear'|'power'
        },
    }

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    pipeline = Pipeline(BASE_DIR, args.output_dir, PREDICTION_PARAMS)
    # --- Step 1: List all PDB files in the folder ---

    total_start_time = time.time()
    per_protein_times = []

    # --- Step 2: Run prediction for each PDB file ---
    pdb_files = [
        f for f in os.listdir(args.output_dir)
        if f.lower().endswith(".pdb")
    ]
    if not pdb_files:
        raise ValueError(f"No PDB files found in output_dir={args.output_dir}")

    pdb_file = pdb_files[0]
    prot_id = os.path.splitext(pdb_file)[0]  # remove ".pdb"

    input_prot_path = os.path.join(args.output_dir, pdb_file)
    csv_path = f"{args.output_dir}/{prot_id}_prediction.csv"

    if os.path.exists(csv_path):
        pass
        #continue

    print("--------------------------------------")
    print(
        f"Starting prediction for protein {prot_id}"
    )
    print("--------------------------------------")

    protein_start_time = time.time()

    if PREDICTION_PARAMS["hier"]:
        print(
            f"*** HIER PREDICTION | {HIER_PARAMS["subg_gen_method"].upper()} {HIER_PARAMS["asa_exposure_percent"]}% Subgraph ***"
        )
        pipeline.hier_prediction(
            prot_id,
            input_prot_path,
            HIER_PARAMS,
        )

    protein_elapsed = time.time() - protein_start_time
    per_protein_times.append(protein_elapsed)

    print(
        f"[✓] Finished prediction for {prot_id}({protein_elapsed:.2f} sec)\n"
    )
    total_elapsed = time.time() - total_start_time
    minutes = total_elapsed // 60
    seconds = total_elapsed % 60

    print(
        f"[✓] All predictions completed. Predictions saved at: {args.output_dir}"
    )
    print(f"[⏱️] Total prediction time: {int(minutes)} min {int(seconds)} sec.")

    if per_protein_times:
        avg_time = sum(per_protein_times) / len(per_protein_times)
        print(f"[⏱️] Average time per prediction: {avg_time:.2f} sec.")
