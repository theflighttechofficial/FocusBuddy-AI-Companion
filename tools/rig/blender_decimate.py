"""Levels of detail (runs with Blender's Python module ``bpy``): decimate a rigged mesh, keeping
its texture coordinates.

    python tools/rig/blender_decimate.py in.npz out.npz

``in.npz``: V (n, 3), F (m, 3), FUV (m, 3, 2) (glTF v-down), target (number of triangles),
optionally lock (n,) bool: vertices that must not move (a seam shared with another part).
``out.npz``: the same arrays for the decimated mesh. Blender's collapse decimation interpolates
the texture coordinates; triangles across a texture seam keep their seam.
"""

import sys

import bpy  # noqa: I001 - bpy first: it registers bmesh
import bmesh
import numpy as np


def main(src: str, dst: str) -> None:
    z = np.load(src)
    V, F, FUV, target = z["V"], z["F"], z["FUV"], int(z["target"])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new("lod")
    me.from_pydata([tuple(v) for v in V.tolist()], [], [tuple(f) for f in F.tolist()])
    me.update()
    uv = me.uv_layers.new(name="UV")
    loop_uv = FUV.reshape(-1, 2).copy()
    loop_uv[:, 1] = 1.0 - loop_uv[:, 1]                       # glTF v-down -> Blender v-up
    # from_pydata keeps the polygon / loop order of F
    uv.data.foreach_set("uv", loop_uv.reshape(-1).astype(np.float32))
    ob = bpy.data.objects.new("lod", me)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = min(1.0, target / max(1, len(F)))
    mod.use_collapse_triangulate = True
    if "lock" in z.files and z["lock"].any():
        # an inverted vertex group: weight 1 = never collapsed
        vg = ob.vertex_groups.new(name="lock")
        vg.add([int(i) for i in np.flatnonzero(z["lock"])], 1.0, "REPLACE")
        mod.vertex_group = "lock"
        mod.invert_vertex_group = True
    dg = bpy.context.evaluated_depsgraph_get()
    me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bm = bmesh.new()
    bm.from_mesh(me2)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    lay = bm.loops.layers.uv.active
    V2 = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    F2 = np.array([[l.vert.index for l in f.loops] for f in bm.faces], dtype=np.int64)
    FUV2 = np.array([[l[lay].uv[:] for l in f.loops] for f in bm.faces], dtype=np.float64)
    FUV2[..., 1] = 1.0 - FUV2[..., 1]
    np.savez_compressed(dst, V=V2, F=F2, FUV=FUV2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
