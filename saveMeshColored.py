import numpy as np
from plyfile import PlyData
import sys
import matplotlib.pyplot as plt
import trimesh


mesh=sys.argv[1]
vmax=0.075

mesh_file=f"{mesh}.ply"
s_file=f"{mesh}_S.txt"
out_glb_mesh_file=f"{mesh}_C.glb"
out_ply_mesh_file=f"{mesh}_C.ply"

values = np.loadtxt(s_file)
vmin= 0.0

# Rescale as Polyscope
rescaled = (values - vmin) / (vmax - vmin)
rescaled = np.clip(rescaled, 0, 1)

plydata = PlyData.read(mesh_file)

x=plydata['vertex'].data['x']
y=plydata['vertex'].data['y']
z=plydata['vertex'].data['z']

vertices=np.stack([x, y, z], axis=-1)
faces = plydata['face'].data['vertex_indices']

f = [np.stack(fa) for fa in faces]
faces=np.array(f)

# === Save the colored mesh ===
cmap = plt.get_cmap('Reds')
colors = cmap(rescaled)[:, :3]  # RGB

mesh_colored = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=(colors * 255).astype(np.uint8))
mesh_colored.export(out_glb_mesh_file)
mesh_colored.export(out_ply_mesh_file)


