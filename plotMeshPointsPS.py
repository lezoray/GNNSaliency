
import numpy as np
import polyscope as ps
from plyfile import PlyData
import sys


mesh=sys.argv[1]
points=sys.argv[2]

points = np.loadtxt(points)
points = points.astype(int)

#display with Polyscope
plydata = PlyData.read(mesh)
nb_vertices=plydata['vertex'].count

x=plydata['vertex'].data['x']
y=plydata['vertex'].data['y']
z=plydata['vertex'].data['z']

x_p=np.take(x,points,axis=0)
y_p=np.take(y,points,axis=0)
z_p=np.take(z,points,axis=0)

vertices=np.stack([x, y, z], axis=-1)
points=np.stack([x_p, y_p, z_p], axis=-1)
faces = plydata['face'].data['vertex_indices']

f = [np.stack(fa) for fa in faces]
faces=np.array(f)

ps.init()
ps.set_ground_plane_mode("none")

# visualize!
ps_mesh = ps.register_surface_mesh("my mesh", vertices, faces, smooth_shade=True)
ps_cloud_opt = ps.register_point_cloud("my points", points,radius=0.007, point_render_mode='sphere', color=[1,0,0])

ps.show()

