# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Manipulation de fichiers et données
import os
import numpy as np

from MeshDataset import MeshDataset
from MeshGNN import MeshGNN

from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Datasets and loaders
mesh_dir="/home/2024002/PARTAGE/Schelling/"
output_dir = os.path.join(mesh_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

print("\n=== Creation of the datasets ===\n")
dataset = MeshDataset(mesh_dir, split='all')
loader = DataLoader(dataset, batch_size=1, shuffle=False)
print("\n=== Creation of the Model ===\n")
model = MeshGNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load best model for testing
model.load_state_dict(torch.load('best_model_FeastConv.pt'))
model.eval()

# === Inference + saving ===
print("\n=== Start of inference ===\n")

with torch.no_grad():
    for i, data in enumerate(tqdm(loader)):
        output = model(data).squeeze().cpu().numpy()  # [num_nodes]
        output = np.clip(output, 0, None) 

        mesh_name = os.path.splitext(dataset.mesh_files[i])[0]  # e.g. "a" from "a.ply"
        output_path = os.path.join(output_dir, f"{mesh_name}_S.txt")

        # Save predicted saliency values
        np.savetxt(output_path, output, fmt="%.6f")

print("\n=== Done. Results saved in: ", output_dir)