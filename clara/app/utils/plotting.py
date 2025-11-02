import plotly.graph_objs as go
import plotly.offline as pyo
import os


def plot_training_loss_curve(loss_history_dict, output_path="training_curve.html"):
    """
    Plots training loss curves for each split using Plotly.

    Args:
        loss_history_dict (dict): Dictionary with keys as split indices and values as lists of losses.
        output_path (str): Path to save the interactive HTML dashboard.
    """
    traces = []
    for split_idx, loss_values in loss_history_dict.items():
        trace = go.Scatter(
            x=list(range(1, len(loss_values) + 1)),
            y=loss_values,
            mode="lines+markers",
            name=f"Split {split_idx}",
            text=[f"{loss:.4f}" for loss in loss_values],  # show 4 decimal places
            textposition="top center",
            hovertemplate="Epoch %{x:.0f}<br>Loss %{y:.4f}<extra></extra>",
        )
        traces.append(trace)

    layout = go.Layout(
        title="Training Loss per Epoch",
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Loss"),
        template="plotly_white",
    )

    fig = go.Figure(data=traces, layout=layout)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pyo.plot(fig, filename=output_path, auto_open=False)
