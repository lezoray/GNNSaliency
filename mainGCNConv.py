
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Manipulation de fichiers et données
import os
import numpy as np
import trimesh
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# PyTorch Geometric
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
# Pour les statistiques
import scipy

from MeshDataset import MeshDataset
from MeshGNN import MeshGNNGCNConv

@torch.no_grad()
def evaluate(model, loader, loss_fn, device='cpu'):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        loss = loss_fn(pred, data.y)
        total_loss += loss.item()
    return total_loss / len(loader)

def train(model, train_loader, val_loader=None, epochs=100, lr=0.001, device='cpu', save_path='best_model_GCN.pt',restart=False):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')

    if restart and val_loader is not None:#restart from the previous best model
        best_val_loss = evaluate(model, val_loader, loss_fn, device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, data.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, loss_fn, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                improved = "*"
            else:
                improved = ""
            print(f"[Epoch {epoch:03}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} {improved}")
        else:
            print(f"[Epoch {epoch:03}] Train Loss: {avg_train_loss:.4f}")



@torch.no_grad()
def test_model(model, loader, device='cpu', name=''):
    model = model.to(device)
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0
    plcc=np.empty(0)
    all_preds = []
    all_targets = []

    for data in loader:
        data = data.to(device)
        pred = model(data)
        loss = loss_fn(pred, data.y)
        total_loss += loss.item()
        all_preds.append(pred.cpu())
        all_targets.append(data.y.cpu())
        result=scipy.stats.pearsonr(data.y,pred)
        plcc=np.append(plcc,result.statistic)
        


    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    mse = np.mean((preds - targets) ** 2)
    mae = mean_absolute_error(targets, preds)
    r2  = r2_score(targets, preds)

    print(f"📊 {name} Set:")
    print(f"PLCC : {np.average(plcc):.6f} ± {np.std(plcc):.6f}")
    print(f"  MSE : {mse:.6f}")
    print(f"  MAE : {mae:.6f}")
    print(f"  R²  : {r2:.6f}")
    print("-" * 30)

    return preds, targets, plcc




# Datasets and loaders
mesh_dir="/home/2024002/PARTAGE/Schelling/"
print("\n=== Creation of the datasets ===\n")
train_dataset = MeshDataset(mesh_dir, split='train')
val_dataset   = MeshDataset(mesh_dir, split='val')
test_dataset  = MeshDataset(mesh_dir, split='test')
print("\n=== Creation of the DataLoader ===\n")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1)
test_loader  = DataLoader(test_dataset, batch_size=1)

print("\n=== Creation of the Model ===\n")
model = MeshGNNGCNConv()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print("\n=== Training + Saving ===\n")
train(model, train_loader, val_loader, epochs=1400, lr=0.001, device=device, save_path='best_model_GCN.pt',restart=False)

#print("\n=== Loading the Model ===\n")
# Load best model (only for testing without training)
#model.load_state_dict(torch.load('best_model_GCN.pt'))
# Evaluate on all splits
print("\n=== Final Evaluation of the Best Model ===\n")
preds1, targets1, plcc1 = test_model(model, train_loader, device=device, name="Train")
preds2, targets2, plcc2 = test_model(model, val_loader,   device=device, name="Validation")
preds3, targets3, plcc3  = test_model(model, test_loader,  device=device, name="Test")

# Concatenate plcc
plcc = np.concatenate((plcc1, plcc2, plcc3))
# Concatenate predictions and targets
preds = np.concatenate((preds1, preds2, preds3))
targets = np.concatenate((targets1, targets2, targets3))
# Display global results
print("\n=== Global Results ===\n")
print(f"PLCC : {np.average(plcc):.6f} ± {np.std(plcc):.6f}")
print(f"  MSE : {np.mean((preds - targets) ** 2):.6f}")
print(f"  MAE : {mean_absolute_error(targets, preds):.6f}")
print(f"  R²  : {r2_score(targets, preds):.6f}")
print("-" * 30)