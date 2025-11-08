import os
import torch
from torch_geometric.data import Dataset, Data
import trimesh
import numpy as np
from sklearn.model_selection import train_test_split


class MeshDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, pre_transform=None, split_ratio=(0.8, 0.1, 0.1), seed=42):
        super().__init__(root=root_dir, transform=transform, pre_transform=pre_transform)
        self.root_dir = root_dir
        self.all_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.ply')])

        # Split
        train_files, valtest_files = train_test_split(self.all_files, test_size=1 - split_ratio[0], random_state=seed)
        val_files, test_files = train_test_split(valtest_files, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=seed)

        if split == 'train':
            self.mesh_files = train_files
        elif split == 'val':
            self.mesh_files = val_files
        elif split == 'test':
            self.mesh_files = test_files
        elif split == 'all':
            self.mesh_files = self.all_files
        else:
            raise ValueError("Split must be 'train', 'val', 'test' or 'all'")

    def normalize_coords(self, pos):
        center = pos.mean(dim=0)
        pos = pos - center  # centrage
        scale = pos.norm(p=2, dim=1).max()
        pos = pos / scale   # mise à l’échelle dans [-1, 1]
        return pos

    def len(self):
        return len(self.mesh_files)

    def get(self, idx):
        mesh_file = self.mesh_files[idx]
        mesh_path = os.path.join(self.root_dir, mesh_file)
        txt_path = mesh_path.replace('.ply', '.val')

        # Charger le maillage
        mesh = trimesh.load(mesh_path, process=False)
        pos = torch.tensor(mesh.vertices, dtype=torch.float)  # [num_nodes, 3]
        pos = self.normalize_coords(pos)

        # Générer les arêtes à partir des faces
        edges = []
        for face in mesh.faces:
            for i in range(3):
                edges.append([face[i], face[(i + 1) % 3]])
                edges.append([face[(i + 1) % 3], face[i]])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        
        # Nom du fichier de courbure
        curv_path = mesh_path.replace('.ply', '_C.txt')

        if os.path.isfile(curv_path):
            # Charger la courbure sauvegardée
            curvature = np.loadtxt(curv_path)
        else:
            # Calculer la courbure moyenne discrète
            curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, radius=0.05)
            curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
            np.savetxt(curv_path, curvature)

        curvature = torch.tensor(curvature, dtype=torch.float).view(-1, 1)
        curvature = (curvature - curvature.mean()) / (curvature.std() + 1e-8)

        # Feature d’entrée : [x, y, z, courbure]
        x = torch.cat([pos, curvature], dim=1)        

        # Charger les valeurs de référence
        y = torch.tensor(np.loadtxt(txt_path), dtype=torch.float).view(-1, 1)

        data = Data(x=x, edge_index=edge_index, y=y)
        return data
