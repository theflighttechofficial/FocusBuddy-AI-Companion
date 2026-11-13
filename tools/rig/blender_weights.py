"""Stage C (runs with ``bpy``): skin weights by Blender's bone heat ("automatic weights").

    python tools/rig/blender_weights.py in.npz out.npz

``in.npz``: V (n, 3) rig-frame vertices, F (m, 3) triangles, names (k,), heads (k, 3), tails (k, 3),
parents (k,) index or -1, deform (k,) bool.  ``out.npz``: W (n, k) weights (0 for non-deforming bones).
"""

import sys

import bpy
import numpy as np


def main(src: str, dst: str) -> None:
    z = np.load(src, allow_pickle=True)
    V, F = z["V"], z["F"]
    names = [str(n) for n in z["names"]]
    heads, tails, parents, deform = z["heads"], z["tails"], z["parents"], z["deform"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new("body")
    me.from_pydata([tuple(map(float, v)) for v in V], [], [tuple(map(int, f)) for f in F])
    me.update()
    ob = bpy.data.objects.new("body", me)
    bpy.context.scene.collection.objects.link(ob)
    arm = bpy.data.armatures.new("rig")
    ao = bpy.data.objects.new("rig", arm)
    bpy.context.scene.collection.objects.link(ao)
    bpy.context.view_layer.objects.active = ao
    ao.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = []
    for i, n in enumerate(names):
        b = arm.edit_bones.new(n)
        b.head = tuple(map(float, heads[i]))
        b.tail = tuple(map(float, tails[i]))
        b.use_deform = bool(deform[i])
        bones.append(b)
    for i, p in enumerate(parents):
        if p >= 0:
            bones[i].parent = bones[int(p)]
            bones[i].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    def heat(target):
        for o in bpy.context.scene.objects:
            o.select_set(False)
        target.select_set(True)
        ao.select_set(True)
        bpy.context.view_layer.objects.active = ao
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")

    def read(target):
        W = np.zeros((len(target.data.vertices), len(names)), np.float32)
        gi = {g.index: names.index(g.name) for g in target.vertex_groups if g.name in names}
        for v in target.data.vertices:
            for g in v.groups:
                if g.group in gi:
                    W[v.index, gi[g.group]] = g.weight
        return W
    heat(ob)
    W = read(ob)
    empty = int((W.sum(1) < 1e-6).sum())
    if empty > 0.02 * len(V):
        # bone heat needs a clean, closed surface: solve it on a voxel remesh of the model and
        # transfer the weights to the real surface (nearest face, interpolated)
        print(f"weights: bone heat failed on {empty} verts - solving on a voxel remesh")
        for g in list(ob.vertex_groups):
            ob.vertex_groups.remove(g)
        ob.parent = None
        for mdf in list(ob.modifiers):
            ob.modifiers.remove(mdf)
        proxy = ob.copy()
        proxy.data = ob.data.copy()
        bpy.context.scene.collection.objects.link(proxy)
        rm = proxy.modifiers.new("remesh", "REMESH")
        rm.mode = "VOXEL"
        rm.voxel_size = float(z["voxel"]) if "voxel" in z else 0.006
        bpy.context.view_layer.objects.active = proxy
        bpy.ops.object.modifier_apply(modifier="remesh")
        heat(proxy)
        Wp = read(proxy)
        print(f"weights: proxy {len(proxy.data.vertices)} verts, {int((Wp.sum(1) < 1e-6).sum())} empty")
        from scipy.spatial import cKDTree
        P = np.array([v.co[:] for v in proxy.data.vertices])
        ok = Wp.sum(1) > 1e-6
        d, idx = cKDTree(P[ok]).query(V, k=4)
        w = 1.0 / np.maximum(d, 1e-6)
        W = (Wp[ok][idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
        empty = int((W.sum(1) < 1e-6).sum())
    print(f"weights: {len(V)} verts, {empty} without weights")
    np.savez_compressed(dst, W=W)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
