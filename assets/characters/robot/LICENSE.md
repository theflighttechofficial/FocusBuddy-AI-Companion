# Robot - 3D model

Reconstructed for FocusBuddy from the Robot character sheets (art supplied by the project owner)
with **TripoSR** (Stability AI and Tripo AI, MIT licence, https://github.com/VAST-AI-Research/TripoSR):
mesh and colour texture by `tools/reconstruct_3d.py` (with the sheet's front view projected onto
the visible surface), the detailed head grafted on by `tools/head_graft.py`, skeleton, skin
weights and export by `tools/build_3d_companions.py`. The facial expressions in `face/` (where
present) are the sheet's own expression panels, lined up with the model's face by
`tools/face_expressions.py`.

The model is an automatic image-to-3D reconstruction, not a hand-modelled asset: the sides the
sheet views do not show (mostly the back) are the network's guess, the surface is softer than a
sculpt, and the skin weights are computed, not painted. Replace `model.glb` with an artist-made
model for final production quality.

TripoSR is MIT licensed (Copyright (c) 2024 Tripo AI & Stability AI). The input artwork belongs
to the project owner.
