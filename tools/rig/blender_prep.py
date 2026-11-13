"""Stage A (runs with Blender's Python module ``bpy``): import a source GLB, weld it, triangulate,
and write the plain mesh data the rigging stages work on.

    python tools/rig/blender_prep.py art/source_glb/fox.glb build/rig/fox_prep.npz build/rig/fox_base.png

Output (Blender frame: +Z up, the model as imported):
    V   (n, 3)     welded vertex positions
    F   (m, 3)     triangles (indices into V)
    FUV (m, 3, 2)  per-corner texture coordinates (glTF convention: v down)
and the base-colour texture as PNG.
"""

import sys

import bpy  # noqa: I001 - bpy first: it registers bmesh
import bmesh
import numpy as np


def main(src: str, out_npz: str, out_png: str) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
    obs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if len(obs) != 1:
        raise SystemExit(f"expected one mesh, found {len(obs)}")
    o = obs[0]
    me = o.data
    me.transform(o.matrix_world)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    uv = bm.loops.layers.uv.active
    V = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    F = np.array([[l.vert.index for l in f.loops] for f in bm.faces], dtype=np.int64)
    FUV = np.array([[l[uv].uv[:] for l in f.loops] for f in bm.faces], dtype=np.float64)
    FUV[..., 1] = 1.0 - FUV[..., 1]                     # Blender v-up -> glTF v-down
    # the base-colour image
    mat = me.materials[0]
    img = None
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            link = n.inputs["Base Color"].links
            if link:
                img = link[0].from_node.image
    if img is None:
        raise SystemExit("no base colour texture")
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1]      # top row first
    np.savez_compressed(out_npz, V=V, F=F, FUV=FUV, tex_size=np.array([h, w]))
    from PIL import Image
    a = (np.clip(px[..., :3], 0, 1) ** (1 / 2.2) if img.colorspace_settings.name != "sRGB" else np.clip(px[..., :3], 0, 1))
    Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(out_png)
    print(f"{src}: {len(V)} verts, {len(F)} tris, texture {w}x{h} ({img.colorspace_settings.name})")


if __name__ == "__main__":
    main(*sys.argv[1:4])
