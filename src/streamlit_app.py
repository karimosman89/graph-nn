import streamlit as st
import torch
from torch_geometric.data import Data
from gnn_model import GNN
import numpy as np

# Load trained GNN model
model = torch.load("gnn_model.pth")
model.eval()

st.title("Social Network Analysis with Graph Neural Networks")

# Input for the user to provide node features
user_input = st.text_area("Enter node features (comma-separated):")

if st.button("Analyze"):
    if user_input:
        features = np.array(user_input.split(',')).astype(float)
        data = Data(x=torch.tensor([features], dtype=torch.float))
        
        with torch.no_grad():
            prediction = model(data)
            st.write("Prediction:", prediction)
    else:
        st.write("Please enter node features.")

