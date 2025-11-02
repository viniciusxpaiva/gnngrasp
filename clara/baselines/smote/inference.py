import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models import Sage_En, Sage_Classifier, Decoder
from data_load import load_input_protein
import argparse


def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhid", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model", type=str, default="sage")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args([])


args = get_default_args()
device = torch.device(
    f"cuda:{args.device}" if args.cuda and torch.cuda.is_available() else "cpu"
)

input_dir = "data/test_input"
checkpoint_dir = "checkpoint/protein"
output_dir = "data/predictions"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if not file.endswith("_inference_data.npz"):
        continue

    protein_id = file.replace("_inference_data.npz", "")
    npz_path = os.path.join(input_dir, file)
    ckpt_name = "newG_cls_2000_False_0.5.pth"
    ckpt_path = os.path.join(checkpoint_dir, protein_id, ckpt_name)

    print(f"[•] Rodando inferência para: {protein_id}")

    # 1. Dados
    adj, features, labels = load_input_protein(npz_path)
    features, adj = features.to(device), adj.to(device)

    # 2. Modelo
    encoder = Sage_En(
        nfeat=features.shape[1], nhid=args.nhid, nembed=args.nhid, dropout=args.dropout
    ).to(device)
    classifier = Sage_Classifier(
        nembed=args.nhid, nhid=args.nhid, nclass=2, dropout=args.dropout
    ).to(device)
    decoder = Decoder(nembed=args.nhid, dropout=args.dropout).to(device)

    # 3. Pesos
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    classifier.load_state_dict(ckpt["classifier"])
    decoder.load_state_dict(ckpt["decoder"])

    encoder.eval()
    classifier.eval()

    # 4. Inferência
    with torch.no_grad():
        embed = encoder(features, adj)
        output = classifier(embed, adj)
        pred = output.argmax(dim=1).cpu().numpy()

    # 5. Salvar CSV
    df = pd.DataFrame({"node_index": np.arange(len(pred)), "predicted_label": pred})
    out_csv = os.path.join(output_dir, f"{protein_id}_predictions.csv")
    df.to_csv(out_csv, index=False)
    print(f"[✓] Predições salvas em: {out_csv}\n")
