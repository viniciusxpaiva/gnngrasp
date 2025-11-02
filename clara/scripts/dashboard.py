import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import os

# Carregar dados de predição (exemplo fictício, substitua com seu caminho)
data_path = "../output/3i0d_A/ensemble_embd_3i0dA_prediction.csv"
data = pd.read_csv(data_path)

# Suponha que o DataFrame tenha as seguintes colunas:
# 'residue_id', 'predicted_label', 'true_label', 'probability', 'protein_id'


def compute_metrics(df):
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    precision = precision_score(df.true_label, df.predicted_label)
    recall = recall_score(df.true_label, df.predicted_label)
    f1 = f1_score(df.true_label, df.predicted_label)
    mcc = matthews_corrcoef(df.true_label, df.predicted_label)
    auc = roc_auc_score(df.true_label, df.probability)
    return precision, recall, f1, mcc, auc


precision, recall, f1, mcc, auc = compute_metrics(data)

# Curva ROC
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(data.true_label, data.probability)

# Curva Precision-Recall
from sklearn.metrics import precision_recall_curve

prec_curve, rec_curve, _ = precision_recall_curve(data.true_label, data.probability)

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Protein Binding Site Prediction Dashboard"),
        html.Div(
            [
                html.H4("Global Metrics"),
                html.Ul(
                    [
                        html.Li(f"Precision: {precision:.4f}"),
                        html.Li(f"Recall: {recall:.4f}"),
                        html.Li(f"F1-score: {f1:.4f}"),
                        html.Li(f"MCC: {mcc:.4f}"),
                        html.Li(f"AUC: {auc:.4f}"),
                    ]
                ),
            ],
            style={"marginBottom": 30},
        ),
        html.Div(
            [
                dcc.Graph(
                    figure=px.histogram(
                        data,
                        x="probability",
                        nbins=50,
                        title="Distribution of Predicted Probabilities",
                    )
                )
            ]
        ),
        html.Div(
            [
                dcc.Graph(
                    figure={
                        "data": [
                            {"x": fpr, "y": tpr, "type": "line", "name": "ROC Curve"},
                        ],
                        "layout": {
                            "title": "ROC Curve",
                            "xaxis": {"title": "FPR"},
                            "yaxis": {"title": "TPR"},
                        },
                    }
                )
            ]
        ),
        html.Div(
            [
                dcc.Graph(
                    figure={
                        "data": [
                            {
                                "x": rec_curve,
                                "y": prec_curve,
                                "type": "line",
                                "name": "PR Curve",
                            },
                        ],
                        "layout": {
                            "title": "Precision-Recall Curve",
                            "xaxis": {"title": "Recall"},
                            "yaxis": {"title": "Precision"},
                        },
                    }
                )
            ]
        ),
        html.Div(
            [
                dcc.Graph(
                    figure=px.histogram(
                        data,
                        x="protein_id",
                        title="Binding Site Count by Protein",
                        color="predicted_label",
                        barmode="group",
                    )
                )
            ]
        ),
        html.Div(
            [
                html.H4("Raw Predictions Table"),
                dash_table.DataTable(
                    data=data.to_dict("records"),
                    columns=[{"name": i, "id": i} for i in data.columns],
                    page_size=10,
                    style_table={"overflowX": "auto"},
                ),
            ]
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
