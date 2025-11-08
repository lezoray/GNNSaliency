
import numpy as np
import polyscope as ps
from plyfile import PlyData
import sys


mesh=sys.argv[1]

mesh_file=f"{mesh}.ply"
gt_file=f"{mesh}.val"
s_file=f"{mesh}_S.txt"

data_gt= np.loadtxt(gt_file)
data_saliency = np.loadtxt(s_file)

#display with Polyscope

plydata = PlyData.read(mesh_file)

x=plydata['vertex'].data['x']
y=plydata['vertex'].data['y']
z=plydata['vertex'].data['z']

vertices=np.stack([x, y, z], axis=-1)
faces = plydata['face'].data['vertex_indices']

f = [np.stack(fa) for fa in faces]
faces=np.array(f)

ps.init()
ps.set_ground_plane_mode("none")

# visualize!
ps_mesh = ps.register_surface_mesh("my mesh", vertices, faces, smooth_shade=True)

ps.get_surface_mesh("my mesh").add_scalar_quantity("Ground truth", data_gt, defined_on='vertices', cmap='reds', vminmax=[0,0.25], enabled=True)
ps.get_surface_mesh("my mesh").add_scalar_quantity("prediction", data_saliency, defined_on='vertices', cmap='reds', vminmax=[0,0.05], enabled=True)

ps.show()
