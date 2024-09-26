import torch
from torch_geometric.data import DataLoader
from gnn_model import GNN
from dataset import SocialNetworkDataset  # Assume you have a custom dataset class
from sklearn.model_selection import train_test_split

# Hyperparameters
num_features = 34  # Example feature size
num_classes = 2  # Example class size
num_epochs = 100
batch_size = 32

# Load and split data
data = SocialNetworkDataset('data/social_network_data.csv')
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = GNN(num_features, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

