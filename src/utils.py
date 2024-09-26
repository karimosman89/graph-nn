import torch
from torch_geometric.data import Data

def create_graph_data(edge_index, node_features):
    """Create a PyTorch Geometric Data object for the GNN."""
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def load_gnn_model(model_path):
    """Load the trained GNN model."""
    return torch.load(model_path)

def predict_with_gnn(model, data):
    """Make predictions using the GNN model."""
    model.eval()
    with torch.no_grad():
        return model(data)

