import torch


def create_debug_graph(device):
    """
    Create a small example graph for debugging the GNN internals.

    Args:
        device (torch.device): Device where tensors should be created (CPU or GPU).

    Returns:
        Tuple: (x_input, edge_index_input, edge_features_input, batch)
    """
    x_input = torch.tensor(
        [
            [0.1, 0.2, 0.3],  # Node 0
            [0.4, 0.5, 0.6],  # Node 1
            [0.7, 0.8, 0.9],  # Node 2
            [1.0, 1.1, 1.2],  # Node 3
        ],
        dtype=torch.float,
        device=device,
    )

    edge_index_input = torch.tensor(
        [
            [0, 1, 1, 2, 1, 3, 2, 3],  # sources
            [1, 0, 2, 1, 3, 1, 3, 2],  # targets
        ],
        dtype=torch.long,
        device=device,
    )

    edge_features_input = torch.tensor(
        [
            [3.1, 0, 1],  # 0 → 1
            [3.1, 0, 1],  # 1 → 0
            [2.5, 1, 0],  # 1 → 2
            [2.5, 1, 0],  # 2 → 1
            [2.9, 0, 1],  # 1 → 3
            [2.9, 0, 1],  # 3 → 1
            [4.0, 1, 0],  # 2 → 3
            [4.0, 1, 1],  # 3 → 2
        ],
        dtype=torch.float,
        device=device,
    )

    batch = torch.zeros(x_input.size(0), dtype=torch.long, device=device)

    return x_input, edge_index_input, edge_features_input, batch
