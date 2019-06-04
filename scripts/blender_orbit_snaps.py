import bpy
import mathutils
import math
import numpy as np

# run as: blender --background --python blender_orbit_snaps.py

# set up output resolution and background color
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.world.horizon_color = (0.,0.,0.)

# camera
# delete any existing
if len(bpy.data.cameras)>0:
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete(use_global=False)

    for item in bpy.data.cameras:
        bpy.data.cameras.remove(item)

# add a new one pointing at the stream y-z coordinates (y being on the y axis, and z on the x axis)
bpy.ops.object.camera_add(location=(20,2,3), rotation=(0.5*np.pi,0,0.5*np.pi))
bpy.context.object.data.type = 'ORTHO'
bpy.context.object.data.ortho_scale = 60
bpy.context.scene.camera = bpy.data.objects['Camera']

# delete any existing objects
if len(bpy.data.meshes)>0:
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete(use_global=False)

    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)

# set up star sizes
N = 2000
np.random.seed(3658)
#s = np.random.pareto(100,size=N) * 3
s = np.random.randn(N)*0.01 + 0.05

# show three snapshots from encounter to today
for snap in [0,4,9]:
    d = np.load('/home/ana/projects/gd1_spur/data/blender_vis/encounter_{:03d}.npz'.format(snap))
    r = np.array([d['x'], d['y'], d['z']])[:,:N]

    for i in range(N):
        verts = [r[:,i]]
        mymesh = bpy.data.meshes.new("Vertex")
        myobject = bpy.data.objects.new("Vertex", mymesh)

        myobject.location = [0,0,2]
        bpy.context.scene.objects.link(myobject)

        mymesh.from_pydata(verts,[],[])

        mat = bpy.data.materials.new(name="halo")
        myobject.data.materials.append(mat)

        myobject.active_material.type = 'HALO'
        myobject.active_material.diffuse_color = (0.8*snap/10, 0.680578, 0.484275)
        myobject.active_material.halo.size = s[i]
        myobject.active_material.halo.hardness = 100
        myobject.active_material.alpha = 1

# save
bpy.data.scenes['Scene'].render.filepath = '/home/ana/projects/gd1_spur/plots/blender/gd1_snaps_{:04d}.png'.format(N)
bpy.ops.render.render(write_still=True)
