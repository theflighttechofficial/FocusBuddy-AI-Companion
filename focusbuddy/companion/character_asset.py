"""Character asset pipeline: one abstraction over production GLB/GLTF characters, the
procedural 3D models and the 2D / sprite fallback.

    asset = CharacterAsset.load("cat", quality="high")
    asset.kind            "gltf" | "procedural"
    asset.lod("medium")   the mesh for a level of detail (GLB only)

Folder layout (built-in assets ship in ``assets/characters``; user / artist assets in
``%LOCALAPPDATA%/FocusBuddy/characters`` override them)::

    characters/<id>/
        model.glb               required for production mode (the "high" LOD)
        model_low.glb           optional LODs (model_medium.glb, model_ultra.glb ...)
        animations/*.glb        optional extra clips (retargeted by bone name)
        textures/               optional, referenced by .gltf files
        metadata.json           optional, see ``DEFAULT_METADATA``

Coordinate convention for assets (the glTF standard)
----------------------------------------------------
* +Y up, the ground is Y = 0 (the lowest foot / paw touches it in the bind pose)
* the character faces +Z; its right side is -X
* any unit - the loader normalises the bind-pose height to the rig's height
* skinned with a single skin whose joints use the canonical names below (or a
  ``bone_map`` in metadata.json / a well-known naming scheme such as Mixamo)

Internally the companion uses renderer units: +x forward, +y down, +z toward the
viewer's side of the body (the character's right when it walks right), feet at 0.
``ASSET_TO_INTERNAL`` converts between the two.

Canonical skeleton
------------------
Root, Pelvis, Spine, Chest, Neck, Head, Jaw, LeftEar, RightEar,
Left/RightUpperArm, Left/RightLowerArm, Left/RightHand   (front legs of animals)
Left/RightUpperLeg, Left/RightLowerLeg, Left/RightFoot   (hind legs)
Tail0, Tail1, ...                                          (optional chain)
Optional extras follow their parent rigidly (Hair, Antenna, Tag, fingers ...).
"""

from __future__ import annotations

import json
import math
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import gltf

log = logging.getLogger("focusbuddy.companion")

REQUIRED_BONES = ("Root", "Pelvis", "Spine", "Chest", "Neck", "Head",
                  "LeftUpperArm", "LeftLowerArm", "LeftHand", "RightUpperArm", "RightLowerArm", "RightHand",
                  "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "RightUpperLeg", "RightLowerLeg", "RightFoot")
OPTIONAL_BONES = ("Jaw", "LeftEar", "RightEar", "LeftEye", "RightEye", "Muzzle")
ANIMAL_BONES = ("Tail0",)

# rig limb name -> canonical (upper, lower, end)
LIMB_BONES = {
    "quadruped": {"FL": ("LeftUpperArm", "LeftLowerArm", "LeftHand"),
                  "FR": ("RightUpperArm", "RightLowerArm", "RightHand"),
                  "BL": ("LeftUpperLeg", "LeftLowerLeg", "LeftFoot"),
                  "BR": ("RightUpperLeg", "RightLowerLeg", "RightFoot")},
    "biped": {"L": ("LeftUpperLeg", "LeftLowerLeg", "LeftFoot"), "R": ("RightUpperLeg", "RightLowerLeg", "RightFoot"),
              "arm_L": ("LeftUpperArm", "LeftLowerArm", "LeftHand"),
              "arm_R": ("RightUpperArm", "RightLowerArm", "RightHand")},
}

# well-known naming schemes -> canonical
_ALIASES = {
    "root": "Root", "armature": "Root", "hips": "Pelvis", "pelvis": "Pelvis", "spine": "Spine", "spine1": "Chest",
    "spine01": "Spine", "spine02": "Chest", "chest": "Chest", "upperchest": "Chest", "neck": "Neck", "head": "Head",
    "jaw": "Jaw", "lowerjaw": "Jaw",
    "leftarm": "LeftUpperArm", "leftupperarm": "LeftUpperArm", "upperarml": "LeftUpperArm", "upperarmleft": "LeftUpperArm",
    "leftforearm": "LeftLowerArm", "leftlowerarm": "LeftLowerArm", "lowerarml": "LeftLowerArm", "forearml": "LeftLowerArm",
    "lefthand": "LeftHand", "handl": "LeftHand",
    "rightarm": "RightUpperArm", "rightupperarm": "RightUpperArm", "upperarmr": "RightUpperArm",
    "upperarmright": "RightUpperArm", "rightforearm": "RightLowerArm", "rightlowerarm": "RightLowerArm",
    "lowerarmr": "RightLowerArm", "forearmr": "RightLowerArm", "righthand": "RightHand", "handr": "RightHand",
    "leftupleg": "LeftUpperLeg", "leftupperleg": "LeftUpperLeg", "thighl": "LeftUpperLeg", "leftthigh": "LeftUpperLeg",
    "leftleg": "LeftLowerLeg", "leftlowerleg": "LeftLowerLeg", "calfl": "LeftLowerLeg", "shinl": "LeftLowerLeg",
    "leftfoot": "LeftFoot", "footl": "LeftFoot",
    "rightupleg": "RightUpperLeg", "rightupperleg": "RightUpperLeg", "thighr": "RightUpperLeg",
    "rightthigh": "RightUpperLeg", "rightleg": "RightLowerLeg", "rightlowerleg": "RightLowerLeg",
    "calfr": "RightLowerLeg", "shinr": "RightLowerLeg", "rightfoot": "RightFoot", "footr": "RightFoot",
    "leftear": "LeftEar", "earl": "LeftEar", "rightear": "RightEar", "earr": "RightEar",
    "lefteye": "LeftEye", "eyel": "LeftEye", "righteye": "RightEye", "eyer": "RightEye",
    # quadruped naming (front legs are the "arms")
    "frontleftupperleg": "LeftUpperArm", "frontleftlowerleg": "LeftLowerArm", "frontleftfoot": "LeftHand",
    "frontrightupperleg": "RightUpperArm", "frontrightlowerleg": "RightLowerArm", "frontrightfoot": "RightHand",
    "backleftupperleg": "LeftUpperLeg", "backleftlowerleg": "LeftLowerLeg", "backleftfoot": "LeftFoot",
    "backrightupperleg": "RightUpperLeg", "backrightlowerleg": "RightLowerLeg", "backrightfoot": "RightFoot",
}

LOD_NAMES = ("low", "medium", "high", "ultra")
LOD_FILES = {"low": "model_low.glb", "medium": "model_medium.glb", "high": "model.glb", "ultra": "model_ultra.glb"}

DEFAULT_METADATA = {
    "id": "",
    "kind": "quadruped",            # quadruped | biped
    "species": "",
    "production": False,            # True only for finished, artist-made assets
    "description": "",
    "bone_map": {},                 # asset bone name -> canonical name
    "face": {"mode": "decal",       # decal: vector eyes / mouth projected on the head (stylised toy)
                                    # morph: the mesh has its own eyes / mouth driven by morph targets
             "eye_dirs": [[-27.0, 0.02], [27.0, 0.02]], "eye_size": 1.3, "head_radius": 21.0,
             "mouth": [0.98, 0.5, 0.0], "mouth_normal": [1.0, 0.45, 0.0], "whiskers": False, "robot": False},
    "morph_map": {},                # morph target name -> {"param": face/pose parameter, "scale": 1}
    "materials": {},                # material name -> stylised shading type (fur, metal, plastic, cloth, skin, eye)
    "clips": {},                    # action name -> clip name (GLB animation) to play instead of the procedural recipe
    "height": None,                 # optional bind-pose height override (asset units)
}

# Facial / body morph targets understood automatically (ARKit-style and common DCC names).
# value = (face parameter, scale); parameters are computed from the pose every frame
# (see ``render_mesh.MeshRenderer3D.face_params``).
ARKIT_MORPHS = {
    "eyeblinkleft": ("blink_left", 1.0), "eyeblinkright": ("blink_right", 1.0), "eyeblink": ("lid", 1.0),
    "blink": ("lid", 1.0), "blinkleft": ("blink_left", 1.0), "blinkright": ("blink_right", 1.0),
    "jawopen": ("jaw", 1.0), "mouthopen": ("jaw", 1.0),
    "mouthsmile": ("happy", 1.0), "mouthsmileleft": ("happy", 1.0), "mouthsmileright": ("happy", 1.0),
    "smile": ("happy", 1.0), "happy": ("happy", 1.0), "joy": ("happy", 1.0),
    "mouthfrown": ("frown", 1.0), "mouthfrownleft": ("frown", 1.0), "mouthfrownright": ("frown", 1.0),
    "frown": ("frown", 1.0), "sad": ("sad", 1.0), "sorrow": ("sad", 1.0),
    "angry": ("angry", 1.0), "anger": ("angry", 1.0), "surprised": ("surprised", 1.0), "surprise": ("surprised", 1.0),
    "browdown": ("brow_down", 1.0), "browdownleft": ("brow_down", 1.0), "browdownright": ("brow_down", 1.0),
    "browup": ("brow_up", 1.0), "browinnerup": ("brow_up", 1.0), "browouterupleft": ("brow_up", 1.0),
    "browouterupright": ("brow_up", 1.0),
    "cheekpuff": ("cheek_puff", 1.0), "cheeksquintleft": ("lid_low", 1.0), "cheeksquintright": ("lid_low", 1.0),
    "eyesquintleft": ("lid_low", 1.0), "eyesquintright": ("lid_low", 1.0),
    "eyewideleft": ("eye_wide", 1.0), "eyewideright": ("eye_wide", 1.0), "breath": ("breath", 1.0),
    "cheeks": ("cheek", 1.0), "mouthpucker": ("mouth_width", -1.0),
}

# asset (+Y up, +Z forward, right = -X)  ->  internal (+x forward, +y down, +z = right side)
ASSET_TO_INTERNAL = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
INTERNAL_TO_ASSET = ASSET_TO_INTERNAL.T


def canonical_name(name: str, bone_map: Optional[dict] = None) -> Optional[str]:
    if bone_map and name in bone_map:
        return bone_map[name]
    if name in REQUIRED_BONES or name in OPTIONAL_BONES or re.fullmatch(r"Tail\d+", name or ""):
        return name
    key = re.sub(r"^(mixamorig\d*:|armature[_|]|def[-_])", "", (name or "").lower())
    key = re.sub(r"[^a-z0-9]", "", key)
    if key in _ALIASES:
        return _ALIASES[key]
    m = re.fullmatch(r"tail0*(\d+)", key)
    if m:
        return f"Tail{int(m.group(1))}"
    return None


def search_dirs() -> list:
    dirs = []
    try:
        from ..config import ASSETS_DIR, data_dir
        dirs.append(Path(data_dir()) / "characters")
        dirs.append(Path(ASSETS_DIR) / "characters")
    except Exception:  # noqa: BLE001
        dirs.append(Path(__file__).resolve().parent.parent.parent / "assets" / "characters")
    return dirs


def asset_dir(char_id: str) -> Optional[Path]:
    for d in search_dirs():
        p = d / char_id
        if (p / "model.glb").is_file() or (p / "model.gltf").is_file():
            return p
    return None


# --------------------------------------------------------------------- data
@dataclass
class Joint:
    name: str                 # name in the file
    canonical: Optional[str]  # canonical name (None = extra bone that follows its parent)
    parent: int               # index into joints (-1 = none)
    bind_world: np.ndarray    # 4x4 in internal units
    inverse_bind: np.ndarray  # 4x4
    rest_t: Optional[np.ndarray] = None   # rest local translation (asset space, file units)
    rest_r: Optional[np.ndarray] = None   # rest local rotation quaternion (x, y, z, w)
    rest_s: Optional[np.ndarray] = None   # rest local scale
    base: Optional[np.ndarray] = None     # root joints: world matrix of the non-joint ancestors (asset space)
    ibm: Optional[np.ndarray] = None      # inverse bind matrix (asset space)


@dataclass
class MeshLOD:
    positions: np.ndarray     # (N, 3) internal units, bind pose
    normals: np.ndarray       # (N, 3)
    colors: np.ndarray        # (N, 3) linear-ish 0..1
    material: np.ndarray      # (N,) int index into CharacterAsset.shading
    indices: np.ndarray       # (M, 3)
    joints: np.ndarray        # (N, 4) int index into CharacterAsset.joints
    weights: np.ndarray       # (N, 4)
    morphs: dict = field(default_factory=dict)        # name -> (N, 3) position deltas
    morph_normals: dict = field(default_factory=dict)
    uvs: Optional[np.ndarray] = None                  # (N, 2) TEXCOORD_0 (zeros without UVs)
    tangents: Optional[np.ndarray] = None             # (N, 4) internal frame xyz + handedness

    @property
    def triangles(self) -> int:
        return int(len(self.indices))


@dataclass
class MaterialDesc:
    """A glTF PBR material as the renderers use it (stylised shading type + PBR inputs)."""
    key: str
    shading: str = "plastic"                          # fur cloth skin plastic metal eye emissive
    base_color: tuple = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 0.8
    emissive: tuple = (0.0, 0.0, 0.0)
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    unlit: bool = False
    textures: dict = field(default_factory=dict)      # baseColor / normal / metallicRoughness / occlusion / emissive
    normal_scale: float = 1.0
    occlusion_strength: float = 1.0
    pbr: bool = True                                  # metallic / roughness drive the highlight

    def has(self, slot: str) -> bool:
        return slot in self.textures


@dataclass
class Clip:
    name: str
    duration: float
    channels: list            # [(joint index, path, times, values, interpolation)] in asset space
    rest: dict = field(default_factory=dict)   # joint index -> (rest quat, rest translation, rest scale) of the source
    source: str = ""
    morph_channels: list = field(default_factory=list)   # [(target names, times, values (K, n), interpolation)]
    own: bool = False                          # authored on this model (local TRS used directly)


SHADING = ("fur", "plastic", "metal", "cloth", "skin", "eye", "emissive")


class CharacterAsset:
    """A loaded character: skeleton + LOD meshes + materials + clips + metadata."""

    def __init__(self, char_id: str, kind: str, metadata: dict, path: Optional[Path] = None):
        self.id = char_id
        self.kind = kind                      # gltf | procedural
        self.meta = metadata
        self.path = path
        self.joints: list[Joint] = []
        self.lods: dict[str, MeshLOD] = {}
        self.clips: dict[str, Clip] = {}
        self.shading: list[str] = []          # per material index: fur / metal / ...
        self.materials: list[MaterialDesc] = []
        self.scale = 1.0
        self.issues: list[str] = []

    # ------------------------------------------------------------ factory
    @classmethod
    def load(cls, char_id: str, quality: str = "high", directory: Optional[Path] = None,
             rig_height: Optional[float] = None) -> "CharacterAsset":
        """GLB asset if one is installed for ``char_id`` (any LOD), else the procedural description."""
        d = Path(directory) if directory is not None else asset_dir(char_id)
        if d is not None:
            try:
                return cls.from_folder(char_id, d, rig_height=rig_height)
            except Exception as e:  # noqa: BLE001 - a broken asset must never break the companion
                log.warning("Character asset %s could not be loaded (%s) - using the procedural model", d, e)
        return cls(char_id, "procedural", dict(DEFAULT_METADATA, id=char_id))

    @property
    def is_mesh(self) -> bool:
        return self.kind == "gltf" and bool(self.lods)

    @property
    def production(self) -> bool:
        return bool(self.meta.get("production")) and self.is_mesh

    def lod(self, quality: str) -> Optional[MeshLOD]:
        q = {"normal": "medium"}.get(quality, quality)
        order = {"low": ("low", "medium", "high", "ultra"), "medium": ("medium", "high", "low", "ultra"),
                 "high": ("high", "ultra", "medium", "low"), "ultra": ("ultra", "high", "medium", "low")}
        for name in order.get(q, order["high"]):
            if name in self.lods:
                return self.lods[name]
        return None

    def joint_index(self, canonical: str) -> int:
        for i, j in enumerate(self.joints):
            if j.canonical == canonical:
                return i
        return -1

    # ------------------------------------------------------------ loading
    @classmethod
    def from_folder(cls, char_id: str, folder: Path, rig_height: Optional[float] = None,
                    refine: str = "all") -> "CharacterAsset":
        folder = Path(folder)
        meta = dict(DEFAULT_METADATA, id=char_id)
        mf = folder / "metadata.json"
        if mf.is_file():
            try:
                user = json.loads(mf.read_text(encoding="utf-8"))
                face = dict(DEFAULT_METADATA["face"], **(user.get("face") or {}))
                meta.update(user)
                meta["face"] = face
            except (OSError, ValueError) as e:
                log.warning("metadata.json of %s ignored: %s", folder, e)
        main = folder / "model.glb"
        if not main.is_file():
            main = folder / "model.gltf"
        doc = gltf.load(main)
        a = cls(char_id, "gltf", meta, folder)
        a._skeleton(doc, rig_height)
        a.lods["high"] = a._mesh(doc)
        for lod, fname in LOD_FILES.items():
            p = folder / fname
            if lod != "high" and p.is_file():
                try:
                    a.lods[lod] = a._mesh(gltf.load(p))
                except Exception as e:  # noqa: BLE001
                    a.issues.append(f"LOD {lod} not loaded: {e}")
        if refine != "none":
            a._apply_refine(folder, geometry_only=refine == "geometry")
        try:
            grow = float(meta.get("head_grow", 1.0) or 1.0)
        except (TypeError, ValueError):
            grow = 1.0
        if abs(grow - 1.0) > 1e-3:
            a._grow_head(max(0.7, min(2.0, grow)))
        if meta.get("hand_zone"):
            for m in a.lods.values():
                a._clean_arm_weights(m)
        if meta.get("skirt"):
            for m in a.lods.values():
                a._skirt_weights(m)
        a._clips(doc, own=True)
        if meta.get("mocap"):
            # real human movement: the shared motion-capture library retargeted onto this skeleton
            try:
                from .mocap import retarget
                a.clips.update(retarget(a))
            except Exception as e:  # noqa: BLE001 - the procedural rig still animates the model
                a.issues.append(f"mocap clips not loaded: {e}")
        anim_dir = folder / "animations"
        if anim_dir.is_dir():
            for p in sorted(anim_dir.glob("*.gl*")):
                try:
                    a._clips(gltf.load(p))
                except Exception as e:  # noqa: BLE001
                    a.issues.append(f"animation file {p.name} not loaded: {e}")
        return a

    # neck span (0 = Neck joint, 1 = Head joint) over which a grown head blends into the body
    HEAD_GROW_FROM, HEAD_GROW_TO = 0.0, 0.75

    def head_grow_weight(self, pts: np.ndarray) -> np.ndarray:
        """0..1 per point: how much of the head growth applies there (bind pose, internal units).
        Full above ~3/4 of the neck, nothing at the neck's base and below - so shoulders, collar and
        long hair hanging down the back keep their size and the neck tapers smoothly into the head."""
        hi, ni = self.joint_index("Head"), self.joint_index("Neck")
        H = self.joints[hi].bind_world[:3, 3]
        N = self.joints[ni].bind_world[:3, 3]
        ax = H - N
        L = float(np.linalg.norm(ax))
        h = (np.asarray(pts, dtype=float) - N) @ (ax / L) / L
        u = np.clip((h - self.HEAD_GROW_FROM) / (self.HEAD_GROW_TO - self.HEAD_GROW_FROM), 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    def _grow_head(self, k: float) -> None:
        """metadata ``head_grow``: a bigger head (``k`` x) baked into the bind pose.

        The skeleton's head joints scale uniformly about the Head joint (bind matrices, the rest
        translations the clips use, the inverse binds), and so does every vertex above the neck;
        down the neck the growth fades out smoothly. Unlike the runtime ``head_scale`` (which
        follows the skin weights), hair or a collar weighted partly to the head never stretches or
        flares out: what decides is where a vertex is, not what it is weighted to."""
        hi, ni = self.joint_index("Head"), self.joint_index("Neck")
        if hi < 0 or ni < 0 or not self.lods:
            self.issues.append("head_grow needs Head and Neck joints - skipped")
            return
        H = self.joints[hi].bind_world[:3, 3].copy()
        if float(np.linalg.norm(H - self.joints[ni].bind_world[:3, 3])) < 1e-6:
            return
        # 1. joints below the Head joint: uniform scale about it
        kids = [i for i in range(len(self.joints)) if i != hi and self._descends(i, hi)]
        for i in kids:
            j = self.joints[i]
            bw = np.array(j.bind_world, dtype=float, copy=True)
            bw[:3, 3] = H + k * (bw[:3, 3] - H)
            j.bind_world = bw
            j.inverse_bind = np.linalg.inv(bw)
            if j.ibm is not None:
                j.ibm = j.inverse_bind @ self._C        # inverse of the asset-space bind (C^-1 @ bind)
            if j.rest_t is not None:
                j.rest_t = np.asarray(j.rest_t, dtype=float) * k
        # 2. vertices and morph deltas: the same scale, fading out down the neck
        for m in self.lods.values():
            s = 1.0 + (k - 1.0) * self.head_grow_weight(m.positions)
            m.positions = H + (m.positions - H) * s[:, None]
            for name, d in list(m.morphs.items()):
                m.morphs[name] = np.asarray(d, dtype=float) * s[:, None]
        self.head_grown = k

    def _descends(self, i: int, ancestor: int) -> bool:
        p = self.joints[i].parent
        while p >= 0:
            if p == ancestor:
                return True
            p = self.joints[p].parent
        return False

    def _apply_refine(self, folder: Path, geometry_only: bool = False) -> None:
        """Bake-time realism data from ``refine.npz`` (``tools/refine/bake_ao.py`` and
        ``tools/refine/faces.py``). Every part is optional and checked against the mesh it is for;
        a mismatch skips that part (noted in ``issues``) and never stops the companion.

        * ``shape_idx_<lod>`` / ``shape_pos_<lod>`` / ``shape_nrm_<lod>``: re-sculpted vertices
          (the people's chin and jaw, smoothed facial skin) - new bind-pose positions and normals.
        * ``skin_idx_<lod>`` / ``skin_j_<lod>`` / ``skin_w_<lod>``: corrected skin weights (the
          people's jaw follows the head; the neck bends along its length).
        * ``ao_<lod>``: per-vertex ambient occlusion (1 = open sky). It darkens creases, armpits,
          the gap under the chin and between the legs. Normalised so open surfaces keep their exact
          original colour. metadata ``"ao_strength"`` (0..1, default 0.6) tunes it; 0 = off.
        * ``tint_idx_<lod>`` / ``tint_<lod>``: baked shading multipliers (the jaw's own shadow).
        * ``face_rgb`` / ``face_rect`` and ``face_tris_<lod>`` / ``face_verts_<lod>`` /
          ``face_uv_<lod>``: the people's faces re-baked into one continuous, retouched image (2.3).
          The image goes into a free corner of the texture; the face's triangles get their own copy
          of their vertices with texture coordinates into it (the rest of the model is unchanged).
        """
        p = folder / "refine.npz"
        if not p.is_file():
            return
        try:
            data = np.load(p)
        except Exception as e:  # noqa: BLE001 - a broken bake must never stop the companion
            self.issues.append(f"refine.npz not loaded: {e}")
            return
        files = set(data.files)
        for lod, m in self.lods.items():
            k = f"shape_idx_{lod}"
            if k not in files:
                continue
            idx = np.asarray(data[k], dtype=np.int64)
            if len(idx) and idx.max() >= len(m.positions):
                self.issues.append(f"{k} does not fit this mesh - skipped")
                continue
            m.positions = np.array(m.positions, dtype=float, copy=True)
            m.positions[idx] = np.asarray(data[f"shape_pos_{lod}"], dtype=float)
            if f"shape_nrm_{lod}" in files:
                m.normals = np.array(m.normals, dtype=float, copy=True)
                m.normals[idx] = np.asarray(data[f"shape_nrm_{lod}"], dtype=float)
            if f"skin_idx_{lod}" in files:
                si = np.asarray(data[f"skin_idx_{lod}"], dtype=np.int64)
                if len(si) and si.max() < len(m.positions):
                    m.joints = np.array(m.joints, copy=True)
                    m.weights = np.array(m.weights, copy=True)
                    m.joints[si] = np.asarray(data[f"skin_j_{lod}"]).astype(m.joints.dtype)
                    m.weights[si] = np.asarray(data[f"skin_w_{lod}"]).astype(m.weights.dtype)
        if geometry_only:
            return
        strength = float(self.meta.get("ao_strength", 0.6))
        for lod, m in self.lods.items():
            key = f"ao_{lod}"
            if key not in files or strength <= 0.0:
                continue
            ao = np.asarray(data[key], dtype=float)
            if len(ao) != len(m.positions):
                self.issues.append(f"{key} has {len(ao)} values for {len(m.positions)} vertices - skipped")
                continue
            ref = max(float(np.percentile(ao, 80)), 1e-3)
            ao_n = np.clip(ao / ref, 0.0, 1.0)
            factor = np.clip(1.0 - strength * (1.0 - ao_n) ** 1.5, 0.4, 1.0)   # gentle on open-ish areas, firm in creases
            m.colors = m.colors * factor[:, None]
        for lod, m in self.lods.items():
            k = f"tint_idx_{lod}"
            if k in files:
                idx = np.asarray(data[k], dtype=np.int64)
                if len(idx) and idx.max() < len(m.positions):
                    m.colors = np.array(m.colors, dtype=float, copy=True)
                    m.colors[idx] *= np.asarray(data[f"tint_{lod}"], dtype=float)[:, None]
        if any(k.startswith("skinmat_") for k in files):
            self._apply_skin(data, files)
        if "face_rgb" in files and self.meta.get("refined_face", True):
            self._apply_face(data, files)

    def _apply_skin(self, data, files) -> None:
        """``skinmat_<lod>``: the vertices that are bare skin (face, ears, neck, arms, hands) get a
        "skin" twin of their material - the same texture, but soft wrapped light with a warm
        glow where the light turns away, instead of the cloth response."""
        import copy
        mats = [i for i, md in enumerate(self.materials) if "baseColor" in getattr(md, "textures", {})]
        if not mats:
            return
        base = mats[0]
        twin = copy.copy(self.materials[base])
        twin.key = f"{twin.key}_skin"
        twin.shading = "skin"
        twin.textures = self.materials[base].textures      # the very same picture (one upload)
        self.materials.append(twin)
        if len(self.shading) == len(self.materials) - 1:
            self.shading.append("skin")
        idx = len(self.materials) - 1
        for lod, m in self.lods.items():
            k = f"skinmat_{lod}"
            if k not in files:
                continue
            flag = np.asarray(data[k]).astype(bool)
            if len(flag) != len(m.positions):
                self.issues.append(f"{k} does not fit this mesh - skipped")
                continue
            mat = np.array(m.material, copy=True)
            mat[flag & (mat == base)] = idx
            m.material = mat

    def _apply_face(self, data, files) -> None:
        mats = [md for md in self.materials if "baseColor" in getattr(md, "textures", {})]
        if not mats:
            return
        md = mats[0]
        tex = md.textures["baseColor"]
        x, y, w, h = (int(v) for v in data["face_rect"])
        img = np.asarray(data["face_rgb"], dtype=np.uint8)
        if img.shape[:2] != (h, w) or y + h > tex.shape[0] or x + w > tex.shape[1]:
            self.issues.append("face image does not fit the texture - skipped")
            return
        if any(f"face_verts_{lod}" in files and np.asarray(data[f"face_verts_{lod}"]).max(initial=0)
               >= len(m.positions) for lod, m in self.lods.items()):
            self.issues.append("face data does not fit this mesh - skipped")
            return
        tex = np.array(tex, dtype=np.uint8, copy=True)
        tex[y:y + h, x:x + w, :3] = img
        if tex.shape[2] == 4:
            tex[y:y + h, x:x + w, 3] = 255
        old = md.textures.get("baseColor")
        for mo in self.materials:                     # every material that draws this picture
            if getattr(mo, "textures", {}).get("baseColor") is old:
                mo.textures = dict(mo.textures)
                mo.textures["baseColor"] = tex
                mo.tex_version = getattr(mo, "tex_version", 0) + 1
        for lod, m in self.lods.items():
            if f"face_verts_{lod}" not in files:
                continue
            src = np.asarray(data[f"face_verts_{lod}"], dtype=np.int64)
            tris = np.asarray(data[f"face_tris_{lod}"], dtype=np.int64)
            uv = np.asarray(data[f"face_uv_{lod}"], dtype=float)
            n0 = len(m.positions)
            remap = np.full(n0, -1, dtype=np.int64)
            remap[src] = n0 + np.arange(len(src))

            def dup(arr):
                return None if arr is None else np.concatenate([arr, arr[src]], axis=0)
            m.positions, m.normals, m.colors = dup(m.positions), dup(m.normals), dup(m.colors)
            m.material, m.joints, m.weights = dup(m.material), dup(m.joints), dup(m.weights)
            m.tangents = dup(m.tangents)
            m.morphs = {k: dup(v) for k, v in m.morphs.items()}
            m.morph_normals = {k: dup(v) for k, v in m.morph_normals.items()}
            base_uv = m.uvs if m.uvs is not None else np.zeros((n0, 2))
            m.uvs = np.concatenate([base_uv, uv.astype(base_uv.dtype)], axis=0)
            ind = np.array(m.indices, copy=True)
            ind[tris] = remap[ind[tris]]
            m.indices = ind

    def _skirt_weights(self, m: "MeshLOD") -> None:
        """metadata ``"skirt": {"hip": h, "knee": h, "side": w}`` (heights over the floor, internal
        units): a long skirt / saree skinned mostly to the pelvis stays a stiff cone when the legs
        move far (sitting on the floor, lying down) and the linear blend between pelvis and thigh
        crumples it. Below the hips the cloth is moved onto the legs instead - gradually, so the
        waist stays with the pelvis - left / right by side (blended across the middle, so the front
        of the skirt hangs between the legs) and from the thigh onto the shin below the knee."""
        cfg = self.meta.get("skirt") or {}
        try:
            hip = float(cfg["hip"])
            knee = float(cfg["knee"])
            side_w = float(cfg.get("side", 6.0))
            depth = float(cfg.get("depth", 0.6))          # down to here (fraction hip..knee) fully on the legs
            band = float(cfg.get("knee_band", 6.0))
        except (KeyError, TypeError, ValueError):
            return
        ix = {n: self.joint_index(n) for n in ("Pelvis", "Root", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg",
                                                 "RightLowerLeg", "LeftFoot", "RightFoot")}
        if min(ix[n] for n in ("Pelvis", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg")) < 0:
            return
        lower = [j for j in ix.values() if j >= 0]
        P = m.positions
        h = -P[:, 1]
        J = m.joints
        W = m.weights
        wl = (W * np.isin(J, lower)).sum(axis=1)                 # weight on pelvis + legs
        sel = np.flatnonzero((h < hip) & (wl > 0.9))
        if not len(sel):
            return
        # long hair hanging down the back past the hips: it follows both legs equally (the body's
        # mean), not the leg on its side - a stepping leg would flick it out behind
        pel_x = float(self.joints[ix["Pelvis"]].bind_world[0, 3])
        hairness = np.zeros(len(sel))
        tex = self.materials[0].textures.get("baseColor") if self.materials else None
        if tex is not None and m.uvs is not None:
            lum = _sample_texture(tex, m.uvs[sel])[:, :3].mean(axis=1)
            behind = np.clip((pel_x - float(cfg.get("hair_behind", 3.0)) - P[sel, 0]) / 2.0, 0.0, 1.0)
            hairness = np.clip((float(cfg.get("hair_lum", 0.22)) - lum) / 0.08, 0.0, 1.0) * behind

        def ss(x):
            x = np.clip(x, 0.0, 1.0)
            return x * x * (3 - 2 * x)
        hs = h[sel]
        a = ss((hip - hs) / max(1e-6, (hip - knee) * depth))     # pelvis -> legs
        b = ss((knee + band - hs) / max(1e-6, 2 * band)) * (1.0 - hairness)   # thigh -> shin
        z = P[sel, 2]
        zl = float(self.joints[ix["LeftUpperLeg"]].bind_world[2, 3])
        sl = np.clip(0.5 + (-z if zl < 0 else z) / (2 * side_w), 0.0, 1.0)
        sl = ss(sl)
        sl = sl + (0.5 - sl) * hairness
        cand = np.stack([1 - a, a * sl * (1 - b), a * sl * b, a * (1 - sl) * (1 - b), a * (1 - sl) * b], axis=1)
        jid = np.array([ix["Pelvis"], ix["LeftUpperLeg"], ix["LeftLowerLeg"], ix["RightUpperLeg"], ix["RightLowerLeg"]])
        order = np.argsort(-cand, axis=1)[:, :4]
        Wn = np.take_along_axis(cand, order, axis=1)
        Wn /= np.maximum(Wn.sum(axis=1, keepdims=True), 1e-9)
        Jn = jid[order]
        J = J.copy()
        W = W.copy()
        J[sel] = Jn
        W[sel] = Wn
        m.joints = J
        m.weights = W

    def _clean_arm_weights(self, m: "MeshLOD") -> None:
        """metadata ``"hand_zone": {"min_abs_z": z, "y": [y0, y1]}`` (internal units): an image-to-3D
        model whose hands hang against the thighs, with arm bones that end above the hand mesh, got
        the hands skinned to the hips / legs - a raised arm left them behind, stretched into strings.
        Vertices in that zone (beside the body, at hand height) are bound wholly to their side's hand."""
        zone = self.meta.get("hand_zone") or {}
        try:
            zmin = float(zone["min_abs_z"])
            y0, y1 = (float(v) for v in zone["y"])
        except (KeyError, TypeError, ValueError):
            return
        hands = {side: self.joint_index(side + "Hand") for side in ("Left", "Right")}
        if min(hands.values()) < 0:
            return
        P = m.positions
        sel = (np.abs(P[:, 2]) >= zmin) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
        if not sel.any():
            return
        side_z = {s_: float(self.joints[j].bind_world[2, 3]) for s_, j in hands.items()}
        J = m.joints.copy()
        W = m.weights.copy()
        for s_, j in hands.items():
            idx = np.flatnonzero(sel & (np.sign(P[:, 2]) == np.sign(side_z[s_])))
            J[idx] = 0
            J[idx, 0] = j
            W[idx] = 0.0
            W[idx, 0] = 1.0
        # above the shoulders (head, headphones) nothing follows an arm
        arm_j = [k for k, j in enumerate(self.joints)
                 if any(t in (j.canonical or "") for t in ("UpperArm", "LowerArm", "Hand"))]
        sh_y = min(float(self.joints[self.joint_index(n)].bind_world[1, 3])
                   for n in ("LeftUpperArm", "RightUpperArm") if self.joint_index(n) >= 0)
        top = np.flatnonzero(P[:, 1] < sh_y - 1.0)
        # around the shoulders: vertices nearer the neck / head than to any arm bone don't follow it
        names = [j.canonical or "" for j in self.joints]
        segs = [(np.array(self.joints[j.parent].bind_world[:3, 3]), np.array(j.bind_world[:3, 3]),
                 any(t in names[j.parent] for t in ("UpperArm", "LowerArm", "Hand")))
                for j in self.joints if j.parent >= 0 and names[self.joints.index(j)] and names[j.parent]]
        band = np.flatnonzero((P[:, 1] >= sh_y - 1.0) & (P[:, 1] < sh_y + 6.0) & np.isin(J, arm_j).any(axis=1))
        if len(band) and segs:
            A = np.array([q[0] for q in segs])
            B = np.array([q[1] for q in segs])
            is_arm = np.array([q[2] for q in segs])
            AB = B - A
            Q = P[band]
            t = np.clip(np.einsum("nsk,sk->ns", Q[:, None, :] - A[None], AB) / np.maximum((AB * AB).sum(1), 1e-9), 0, 1)
            d = np.linalg.norm(Q[:, None, :] - (A[None] + t[..., None] * AB[None]), axis=2)
            far = np.where(is_arm[None], d, np.inf).min(axis=1) > np.where(~is_arm[None], d, np.inf).min(axis=1) + 1.0
            top = np.concatenate([top, band[far]])
        if len(top):
            wt = W[top] * ~np.isin(J[top], arm_j)
            tot = wt.sum(axis=1, keepdims=True)
            ok = tot[:, 0] > 1e-6
            W[top[ok]] = wt[ok] / tot[ok]
            # only arm weights: the vertex goes to the head (above the shoulders) / chest
            rest = top[~ok]
            if len(rest):
                hd = self.joint_index("Head")
                ch = self.joint_index("Chest") if self.joint_index("Chest") >= 0 else hd
                tgt = np.where(P[rest, 1] < sh_y - 3.0, hd, ch)
                J[rest] = 0
                J[rest, 0] = tgt
                W[rest] = 0.0
                W[rest, 0] = 1.0
        m.joints, m.weights = J, W
        self.issues.append(f"hand weights fixed on {int(sel.sum())} vertices")

    def _smooth_crease(self) -> Optional[float]:
        """metadata.json ``"normals": "smooth"`` (+ optional ``"crease": degrees``) recomputes soft
        normals for faceted models; ``None`` keeps the file's normals."""
        if str(self.meta.get("normals", "")).lower() != "smooth":
            return None
        return float(self.meta.get("crease", 60.0))

    def _skeleton(self, doc: gltf.Document, rig_height: Optional[float]) -> None:
        if not doc.skins:
            if not doc.animations:
                raise gltf.GLTFError("no skin (the model must be rigged)")
            # rigid "node rig" (parts animated by node transforms, e.g. Kenney / blocky toys):
            # every node above a mesh becomes a joint
            synth_skin(doc)
        extend_skin(doc, self.meta.get("bone_map"))
        skin = doc.skins[0]
        parents = doc.parents()
        jset = {n: i for i, n in enumerate(skin.joints)}
        # bind pose height (asset units) from every mesh in its bind / rest placement
        recs = _gather(doc)
        allp = np.concatenate([r["pos"] for r in recs]) if recs else np.zeros((1, 3))
        height = float(self.meta.get("height") or (allp[:, 1].max() - min(0.0, allp[:, 1].min())))
        target = float(self.meta.get("rig_height") or rig_height or
                       (74.0 if self.meta.get("kind", "quadruped") == "quadruped" else 93.0))
        self.scale = target / max(1e-6, height)
        self.ground = float(allp[:, 1].min())
        C = np.eye(4)
        C[:3, :3] = ASSET_TO_INTERNAL * self.scale
        C[1, 3] = self.ground * self.scale          # the asset's ground (lowest point) lands on the feet line
        self._C = C
        bone_map = self.meta.get("bone_map") or {}
        world = doc.world_matrices()
        for ji, node_i in enumerate(skin.joints):
            p = parents[node_i]
            while p is not None and p not in jset:
                p = parents[p]
            # bind = rest pose in world (asset) space; skinned vertices are moved into that space too
            # (see _gather), so rigid bone-parented parts and skinned parts share one bind space
            bw_asset = np.asarray(world[node_i], dtype=float)
            bw = C @ bw_asset
            name = doc.nodes[node_i].name
            nd = doc.nodes[node_i]
            if nd.matrix is not None:
                rt, rr, rs = _decompose(nd.matrix)
            else:
                rt, rr, rs = np.array(nd.translation, float), np.array(nd.rotation, float), np.array(nd.scale, float)
            par = parents[node_i]
            base = None
            if p is None:
                base = np.asarray(world[par], dtype=float) if par is not None else np.eye(4)
            self.joints.append(Joint(name, canonical_name(name, bone_map), jset[p] if p is not None else -1, bw,
                                     np.linalg.inv(bw), rt, rr, rs, base, np.linalg.inv(bw_asset)))
        self._node_to_joint = {n: i for i, n in enumerate(skin.joints)}

    # ------------------------------------------------------------ materials
    def _material(self, doc: gltf.Document, mi: Optional[int]) -> int:
        """Index into ``self.materials`` for a document material (shared across LOD files by name)."""
        if mi is None or not (0 <= mi < len(doc.materials)):
            key = "__default__"
            m = None
        else:
            m = doc.materials[mi]
            key = m.name or f"material{mi}"
        for i, md in enumerate(self.materials):
            if md.key == key:
                return i
        if m is None:
            md = MaterialDesc(key, "plastic")
        else:
            shading = (m.extras or {}).get("shading") or (self.meta.get("materials") or {}).get(m.name) or \
                _guess_shading(m)
            em = np.asarray(m.emissive, dtype=float) * float(m.emissive_strength)
            if em.max() > 0.3 and m.emissive_tex is None and shading not in ("eye",):
                shading = "emissive"
            # glTF factors are linear; the stylised shading works on display (sRGB) colours like textures
            bc = np.asarray(m.base_color, dtype=float)
            bc[:3] = linear_to_srgb(bc[:3])
            em = linear_to_srgb(em) if em.max() > 0 else em
            md = MaterialDesc(key, shading, tuple(float(v) for v in bc), float(m.metallic),
                              float(m.roughness), tuple(float(v) for v in em), m.alpha_mode, float(m.alpha_cutoff),
                              bool(m.double_sided), bool(m.unlit))
            # textures (RGBA8, kept local; the GPU path uploads them once, the CPU path samples per pixel)
            for slot, ref in m.textures().items():
                if ref.ok:
                    md.textures[slot] = ref.image
                    if slot == "normal":
                        md.normal_scale = ref.scale
                    elif slot == "occlusion":
                        md.occlusion_strength = ref.scale
                else:
                    self.issues.append(f"material {m.name!r}: {slot} texture missing ({ref.error or 'unknown'})")
            explicit = bool((m.extras or {}).get("shading") or (self.meta.get("materials") or {}).get(m.name))
            md.pbr = bool(md.textures) or not explicit
            _apply_override(md, (self.meta.get("material_overrides") or {}), m.name)
        self.materials.append(md)
        self.shading = [x.shading for x in self.materials]
        return len(self.materials) - 1

    def _mesh(self, doc: gltf.Document) -> MeshLOD:
        C = self._C
        lin = C[:3, :3]
        nlin = ASSET_TO_INTERNAL
        skin = doc.skins[0] if doc.skins else None
        by_name = {j.name: i for i, j in enumerate(self.joints)}
        parts = []
        for r in _gather(doc, self._smooth_crease()):
            prim, node_i = r["prim"], r["node"]
            n = len(prim.positions)
            pos, nor, idx = r["pos"], r["nrm"], r["idx"]
            col = prim.colors[:, :3].astype(float) if prim.colors is not None else np.ones((n, 3))
            if prim.colors is not None and self.meta.get("vertex_colors") != "display":
                col = linear_to_srgb(col)                     # glTF COLOR_0 is linear
            mi = self._material(doc, prim.material)
            md = self.materials[mi]
            col = col * np.asarray(md.base_color[:3], dtype=float)[None, :]
            uv = prim.uvs.astype(float) if prim.uvs is not None else None
            if uv is not None and prim.material is not None and prim.material < len(doc.materials):
                bt = doc.materials[prim.material].base_tex
                if bt is not None and bt.transform:
                    uv = _uv_transform(uv, bt.transform)
            tan = r["tan"]
            if tan is None and uv is not None:
                tan = _tangents(pos, nor, uv, idx)
            if prim.joints is not None and r["skinned"]:
                jskin = doc.skins[r["skin"]] if r["skin"] is not None and r["skin"] < len(doc.skins) else skin
                names = [doc.nodes[k].name for k in jskin.joints]
                remap = np.array([by_name.get(nm, 0) for nm in names], dtype=np.int64)
                jn = remap[np.clip(prim.joints.astype(np.int64), 0, len(remap) - 1)]
                wt = prim.weights.astype(float)
            else:
                # rigid part parented to a bone (no skin): follows its nearest joint ancestor
                ji = self._rigid_joint(doc, node_i, by_name)
                jn = np.zeros((n, 4), dtype=np.int64)
                jn[:, 0] = ji
                wt = np.zeros((n, 4))
                wt[:, 0] = 1.0
            s_ = wt.sum(axis=1, keepdims=True)
            wt = np.where(s_ > 1e-8, wt / np.maximum(s_, 1e-8), np.array([1.0, 0, 0, 0]))
            morphs, mnorm = {}, {}
            tnames = doc.meshes[r["mesh"]].target_names
            for ti, t in enumerate(prim.targets):
                nm = tnames[ti] if ti < len(tnames) else f"target{ti}"
                R3 = r["lin"]
                if t.get("POSITION") is not None:
                    morphs[nm] = (t["POSITION"].astype(float) @ R3.T) @ lin.T
                if t.get("NORMAL") is not None:
                    mnorm[nm] = (t["NORMAL"].astype(float) @ R3.T) @ nlin.T
            uvs = uv if uv is not None else np.zeros((n, 2))
            # the internal frame is mirrored (det -1): the bitangent sign flips with it
            tn = np.zeros((n, 4)) if tan is None else np.concatenate([tan[:, :3] @ nlin.T, -tan[:, 3:4]], axis=1)
            parts.append((pos @ lin.T + C[:3, 3], nor @ nlin.T, col, np.full(n, mi), idx, jn, wt, morphs, mnorm,
                          uvs, tn))
        if not parts:
            raise gltf.GLTFError("no triangle mesh")
        offs = np.cumsum([0] + [len(p[0]) for p in parts])
        names = sorted({k for p in parts for k in p[7]})
        morphs = {k: np.concatenate([p[7].get(k, np.zeros_like(p[0])) for p in parts]) for k in names}
        mnorm = {k: np.concatenate([p[8].get(k, np.zeros_like(p[0])) for p in parts]) for k in names
                 if any(k in p[8] for p in parts)}
        nrm = np.concatenate([p[1] for p in parts])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        idx = np.concatenate([p[4] + offs[i] for i, p in enumerate(parts)]).astype(np.int64)
        return MeshLOD(np.concatenate([p[0] for p in parts]), nrm, np.clip(np.concatenate([p[2] for p in parts]), 0, 1),
                       np.concatenate([p[3] for p in parts]).astype(np.int32), idx,
                       np.concatenate([p[5] for p in parts]).astype(np.int64),
                       np.concatenate([p[6] for p in parts]), morphs, mnorm,
                       np.concatenate([p[9] for p in parts]).astype(np.float32),
                       np.concatenate([p[10] for p in parts]).astype(np.float32))

    def _rigid_joint(self, doc: gltf.Document, node_i: int, by_name: dict) -> int:
        parents = doc.parents()
        if getattr(doc, "_synthetic_skin", False) and doc.nodes[node_i].name in by_name:
            return by_name[doc.nodes[node_i].name]      # node rig: the mesh node is its own joint
        p = parents[node_i]
        while p is not None:
            ji = by_name.get(doc.nodes[p].name)
            if ji is not None:
                return ji
            p = parents[p]
        return max(0, self.joint_index("Root"), self.joint_index("Pelvis"))

    def _clips(self, doc: gltf.Document, own: bool = False) -> None:
        """Animation clips of ``doc`` mapped onto this skeleton (by bone name, then canonical name).
        ``own`` = the clip was authored on this very model (local TRS can be used as-is)."""
        by_name = {j.name: i for i, j in enumerate(self.joints)}
        by_canon = {j.canonical: i for i, j in enumerate(self.joints) if j.canonical}
        bone_map = self.meta.get("bone_map") or {}

        def joint_of(name):
            ji = by_name.get(name)
            if ji is None:
                ji = by_canon.get(canonical_name(name, bone_map))
            return ji
        for a in doc.animations:
            chans, morph = [], []
            for c in a.channels:
                if c.path == "weights":
                    nd = doc.nodes[c.node]
                    if nd.mesh is not None:
                        names = doc.meshes[nd.mesh].target_names or \
                            [f"target{k}" for k in range(len(doc.meshes[nd.mesh].primitives[0].targets))]
                        morph.append((names, c.times, c.values, c.interpolation))
                    continue
                if c.path not in ("rotation", "translation", "scale"):
                    continue
                ji = joint_of(doc.nodes[c.node].name)
                if ji is None:
                    continue
                chans.append((ji, c.path, c.times, c.values, c.interpolation))
            if chans or morph:
                rest = {}
                for n in doc.nodes:
                    ji = joint_of(n.name)
                    if ji is not None:
                        rest[ji] = (np.array(n.rotation, dtype=float), np.array(n.translation, dtype=float),
                                    np.array(n.scale, dtype=float))
                clip = Clip(a.name, a.duration, chans, rest, doc.source)
                clip.morph_channels = morph
                clip.own = own
                self.clips[a.name] = clip

def _hex_rgb(v) -> tuple:
    v = str(v).lstrip("#")
    return tuple(int(v[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _apply_override(md: "MaterialDesc", overrides: dict, name: Optional[str]) -> None:
    """metadata.json ``material_overrides``: {"<material name>" | "*": {"color": "#RRGGBB" (display
    colour), "shading": "fur", "metallic": 0, "roughness": 0.8, "pbr": false}}. Lets a CC0 model be
    recoloured or re-shaded without editing the GLB ("*" applies to every material first)."""
    for key in ("*", name):
        o = overrides.get(key) if key is not None else None
        if not o:
            continue
        if "color" in o:
            r, g, b = _hex_rgb(o["color"])
            md.base_color = (r, g, b, md.base_color[3] if len(md.base_color) > 3 else 1.0)
        if "shading" in o:
            md.shading = str(o["shading"])
            md.pbr = bool(md.textures)
        if "metallic" in o:
            md.metallic = float(o["metallic"])
        if "roughness" in o:
            md.roughness = float(o["roughness"])
        if "pbr" in o:
            md.pbr = bool(o["pbr"])


def _guess_shading(m: gltf.Material) -> str:
    n = (m.name or "").lower()
    for key, words in (("eye", ("eye", "pupil", "iris")), ("metal", ("metal", "chrome", "steel")),
                       ("cloth", ("cloth", "shirt", "hoodie", "pants", "jeans", "fabric")),
                       ("skin", ("skin", "face", "hand")), ("fur", ("fur", "hair", "coat"))):
        if any(w in n for w in words):
            return key
    if m.metallic > 0.6:
        return "metal"
    return "plastic" if m.roughness < 0.45 else "fur"


def _vertex_normals(pos: np.ndarray, idx: np.ndarray) -> np.ndarray:
    v0, v1, v2 = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    n = np.zeros_like(pos)
    for k in range(3):
        np.add.at(n, idx[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def _sample_texture(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest sample (0..1 floats) - used by tools and the validator previews."""
    h, w = tex.shape[:2]
    u = np.mod(uv[:, 0], 1.0) * (w - 1)
    v = np.mod(uv[:, 1], 1.0) * (h - 1)
    t = tex[v.astype(int), u.astype(int), :3]
    return t.astype(np.float32) / 255.0 if t.dtype == np.uint8 else t


def synth_skin(doc: gltf.Document) -> None:
    """Give an unskinned, node-animated model a skin: joints = every node on the path from the
    scene roots to a mesh, inverse bind = inverse rest world matrix."""
    if doc.skins:
        return
    parents = doc.parents()
    keep = set()
    for i, n in enumerate(doc.nodes):
        if n.mesh is not None:
            j = i
            while j is not None:
                keep.add(j)
                j = parents[j]
    animated = {c.node for a in doc.animations for c in a.channels}
    keep |= animated
    order = []
    seen = set()

    def visit(i):
        if i in seen:
            return
        seen.add(i)
        if i in keep:
            order.append(i)
        for c in doc.nodes[i].children:
            visit(c)
    for r in doc.scene_roots:
        visit(r)
    world = doc.world_matrices()
    ibm = np.array([np.linalg.inv(np.asarray(world[j], dtype=float)) for j in order])
    doc.skins.append(gltf.Skin("synthetic", order, ibm, order[0] if order else None))
    doc._synthetic_skin = True


def extend_skin(doc: gltf.Document, bone_map: Optional[dict]) -> None:
    """Add bone-map targets that are plain nodes below the skeleton (e.g. Blender ``*_end`` leaf
    nodes) as extra, weightless joints. Rigs whose leg IK targets live outside the leg chain
    (Quaternius animals) map their feet to the chain's ``_end`` node this way, so two-bone IK
    gets a real upper -> lower -> end chain."""
    if not doc.skins or not bone_map:
        return
    skin = doc.skins[0]
    jset = set(skin.joints)
    parents = doc.parents()
    joint_names = {doc.nodes[j].name for j in skin.joints}
    by_name = {n.name: i for i, n in enumerate(doc.nodes)}
    world = doc.world_matrices()
    extra = []
    for name in bone_map:
        i = by_name.get(name)
        # names already used by a joint stay that joint (mesh nodes may share a bone's name)
        if i is None or i in jset or name in joint_names:
            continue
        p = parents[i]
        while p is not None and p not in jset:
            p = parents[p]
        if p is not None:
            extra.append(i)
    if extra:
        skin.joints = list(skin.joints) + extra
        ibm = np.array([np.linalg.inv(np.asarray(world[j], dtype=float)) for j in extra])
        skin.inverse_bind = np.concatenate([np.asarray(skin.inverse_bind, dtype=float).reshape(-1, 4, 4), ibm])


def _gather(doc: gltf.Document, smooth: Optional[float] = None) -> list:
    """Every triangle primitive in its bind / rest placement (asset space), with welded smooth
    normals when the file has none and winding fixed for mirrored nodes."""
    out = []
    world = doc.world_matrices()
    for node_i, node in enumerate(doc.nodes):
        if node.mesh is None or node.mesh >= len(doc.meshes):
            continue
        for prim in doc.meshes[node.mesh].primitives:
            skinned = node.skin is not None and prim.joints is not None and prim.weights is not None and \
                node.skin < len(doc.skins)
            pos = prim.positions.astype(float)
            idx = np.asarray(prim.indices, dtype=np.int64).reshape(-1, 3)
            if skinned:
                # rest-pose skinning into world space: sum_k w_k * (W_rest(joint_k) @ IBM_k) @ v
                sk = doc.skins[node.skin]
                Jm = np.array([np.asarray(world[j], dtype=float) @ sk.inverse_bind[k] for k, j in enumerate(sk.joints)])
                jj = np.clip(prim.joints.astype(np.int64), 0, len(Jm) - 1)
                ww = prim.weights.astype(float)
                ww = ww / np.maximum(ww.sum(axis=1, keepdims=True), 1e-9)
                Mv = np.einsum("nk,nkij->nij", ww, Jm[jj])
                Wm = np.eye(4)
                Lv = Mv[:, :3, :3]
                pos = np.einsum("nij,nj->ni", Lv, pos) + Mv[:, :3, 3]
            else:
                Mv = None
                Wm = np.asarray(world[node_i], dtype=float)
                pos = pos @ Wm[:3, :3].T + Wm[:3, 3]
            L = Wm[:3, :3] if Mv is None else np.mean(Mv[:, :3, :3], axis=0)
            if np.linalg.det(L) < 0:
                idx = idx[:, ::-1].copy()                 # mirrored node: keep counter-clockwise front faces
            Ln = np.linalg.inv(L).T if abs(np.linalg.det(L)) > 1e-12 else np.eye(3)
            if prim.normals is not None and smooth is None:
                if Mv is not None:
                    nor = np.einsum("nij,nj->ni", np.linalg.inv(Mv[:, :3, :3]).transpose(0, 2, 1),
                                    prim.normals.astype(float))
                else:
                    nor = prim.normals.astype(float) @ Ln.T
                nor /= np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-9)
                # inside-out guard: geometric normals that mostly disagree with the vertex normals
                fn = _face_normals(pos, idx)
                agree = (fn * nor[idx].mean(axis=1)).sum(axis=1)
                if len(agree) and float((agree < 0).mean()) > 0.6:
                    idx = idx[:, ::-1].copy()
            else:
                # no normals, or metadata "normals": "smooth": welded, crease-aware smooth normals
                # (turns faceted low-poly shading into soft shading; hard edges above the crease stay)
                nor = _welded_normals(pos, idx, smooth if smooth is not None else 70.0)
            tan = None
            if prim.tangents is not None:
                t3 = np.einsum("nij,nj->ni", Mv[:, :3, :3], prim.tangents[:, :3].astype(float)) if Mv is not None \
                    else prim.tangents[:, :3].astype(float) @ L.T
                tan = np.concatenate([t3, prim.tangents[:, 3:4]], axis=1)
            out.append({"prim": prim, "node": node_i, "mesh": node.mesh, "skin": node.skin, "skinned": skinned,
                        "pos": pos, "nrm": nor, "idx": idx, "tan": tan, "lin": L})
    return out


def linear_to_srgb(c) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=float), 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.minimum(c, 1.0), 1 / 2.4) - 0.055)


def _decompose(m: np.ndarray):
    m = np.asarray(m, dtype=float)
    t = m[:3, 3].copy()
    sc = np.linalg.norm(m[:3, :3], axis=0)
    R = m[:3, :3] / np.where(sc > 1e-12, sc, 1.0)[None, :]
    if np.linalg.det(R) < 0:
        sc[0] = -sc[0]
        R[:, 0] = -R[:, 0]
    return t, gltf.mat_to_quat(R), sc


def _face_normals(pos: np.ndarray, idx: np.ndarray) -> np.ndarray:
    v0, v1, v2 = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    return fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)


def _welded_normals(pos: np.ndarray, idx: np.ndarray, crease_deg: float = 70.0) -> np.ndarray:
    """Vertex normals for meshes that come without them: corners that share a position are welded
    (no seams at UV splits / in unindexed meshes) and each corner averages the area-weighted
    normals of the faces around that position that bend less than ``crease_deg`` from its own
    face - smooth surfaces stay smooth, hard edges stay hard."""
    v0, v1, v2 = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)                         # area weighted
    fnu = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    span = float(np.ptp(pos, axis=0).max()) or 1.0
    key = np.round(pos / (span * 1e-5)).astype(np.int64)
    _, weld = np.unique(key, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    corner_v = idx.reshape(-1)                             # corner -> vertex
    corner_f = np.repeat(np.arange(len(idx)), 3)           # corner -> face
    cw = weld[corner_v]
    order = np.argsort(cw, kind="stable")
    cw_s = cw[order]
    starts = np.flatnonzero(np.r_[True, cw_s[1:] != cw_s[:-1]])
    ends = np.r_[starts[1:], len(cw_s)]
    cos_c = math.cos(math.radians(crease_deg))
    out = np.zeros_like(pos)
    for a, b in zip(starts, ends):
        cs = order[a:b]
        faces = corner_f[cs]
        Nu = fnu[faces]
        Na = fn[faces]
        if len(cs) == 1:
            out[corner_v[cs]] += Na
            continue
        ok = (Nu @ Nu.T) >= cos_c                         # (k, k): faces within the crease angle
        acc = ok.astype(float) @ Na
        np.add.at(out, corner_v[cs], acc)
    bad = np.linalg.norm(out, axis=1) < 1e-12
    if bad.any():
        out[bad] = np.array([0.0, 1.0, 0.0])
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _tangents(pos: np.ndarray, nor: np.ndarray, uv: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per-vertex tangents (xyz, handedness w) from UVs - Lengyel's method, Gram-Schmidt."""
    p0, p1, p2 = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    w0, w1, w2 = uv[idx[:, 0]], uv[idx[:, 1]], uv[idx[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = w1 - w0, w2 - w0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    ok = np.abs(r) > 1e-12                                  # degenerate UV triangles contribute nothing
    inv = np.where(ok, 1.0 / np.where(ok, r, 1.0), 0.0)
    sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * inv[:, None]
    tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * inv[:, None]
    t = np.zeros_like(pos)
    b = np.zeros_like(pos)
    for k in range(3):
        np.add.at(t, idx[:, k], sdir)
        np.add.at(b, idx[:, k], tdir)
    t = t - nor * (nor * t).sum(axis=1, keepdims=True)
    ln = np.linalg.norm(t, axis=1, keepdims=True)
    axis = np.where(np.abs(nor[:, 1:2]) > 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    fallback = np.cross(nor, axis)
    t = np.where(ln > 1e-9, t / np.maximum(ln, 1e-12), fallback)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    w = np.where((np.cross(nor, t) * b).sum(axis=1) < 0.0, -1.0, 1.0)
    return np.concatenate([t, w[:, None]], axis=1)


def _uv_transform(uv: np.ndarray, tr: dict) -> np.ndarray:
    """KHR_texture_transform: offset, rotation, scale (applied to the vertex UVs)."""
    off = np.asarray(tr.get("offset", (0.0, 0.0)), dtype=float)
    sc = np.asarray(tr.get("scale", (1.0, 1.0)), dtype=float)
    rot = float(tr.get("rotation", 0.0))
    c, s_ = np.cos(rot), np.sin(rot)
    u = uv * sc[None, :]
    return np.stack([c * u[:, 0] + s_ * u[:, 1], -s_ * u[:, 0] + c * u[:, 1]], axis=1) + off[None, :]


# ------------------------------------------------------------------ validator
@dataclass
class Issue:
    level: str        # error | warning | info
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.message}"


@dataclass
class Check:
    name: str
    status: str               # ok | warn | fail | info
    detail: str = ""

    @property
    def mark(self) -> str:
        return {"ok": "\u2713", "warn": "!", "fail": "\u2717", "info": "-"}[self.status]

    def __str__(self) -> str:
        return f"{self.mark} {self.name}" + (f": {self.detail}" if self.detail else "")


def validate(path, kind: Optional[str] = None) -> list[Issue]:
    """Issue list (errors / warnings / info) - see ``check`` for the per-topic report."""
    return check(path, kind)[1]


def check(path, kind: Optional[str] = None) -> tuple[list[Check], list[Issue]]:
    """Check a GLB / GLTF character: mesh, skeleton, required bones, materials, textures (and missing
    ones), animations, morph targets, scale, ground alignment and orientation (faces +Z, Y up)."""
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    model = path if path.is_file() else (folder / "model.glb" if (folder / "model.glb").is_file() else folder / "model.gltf")
    out: list[Issue] = []
    checks: list[Check] = []

    def add(name, status, detail=""):
        checks.append(Check(name, status, detail))
    if not model.is_file():
        add("Mesh", "fail", f"no model.glb / model.gltf in {folder}")
        return checks, [Issue("error", f"no model.glb / model.gltf in {folder}")]
    meta = {}
    if (folder / "metadata.json").is_file():
        try:
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        except ValueError as e:
            out.append(Issue("error", f"metadata.json is not valid JSON: {e}"))
            add("Metadata", "fail", f"metadata.json is not valid JSON: {e}")
    else:
        out.append(Issue("warning", "no metadata.json (defaults are used: quadruped, decal face)"))
        add("Metadata", "warn", "no metadata.json - defaults are used (quadruped, decal face)")
    kind = kind or meta.get("kind", "quadruped")
    try:
        doc = gltf.load(model)
    except Exception as e:  # noqa: BLE001
        add("Mesh", "fail", f"cannot read {model.name}: {e}")
        return checks, out + [Issue("error", f"cannot read {model.name}: {e}")]
    # ---- mesh
    meshes = [n for n in doc.nodes if n.mesh is not None]
    tri = sum(len(p.indices) for n in meshes for p in doc.meshes[n.mesh].primitives)
    verts = sum(len(p.positions) for n in meshes for p in doc.meshes[n.mesh].primitives)
    if not meshes:
        out.append(Issue("error", "no mesh"))
        add("Mesh", "fail", "no mesh")
    else:
        out.append(Issue("info", f"{len(meshes)} mesh node(s), {tri} triangles"))
        nn = sum(1 for n in meshes for p in doc.meshes[n.mesh].primitives if p.normals is None)
        add("Mesh", "warn" if tri > 120_000 else "ok", f"{len(meshes)} mesh node(s), {verts} vertices, {tri} triangles"
            + (", normals computed for %d primitive(s)" % nn if nn else ""))
    if tri > 120_000:
        out.append(Issue("warning", f"{tri} triangles is heavy for a desktop pet - add model_low.glb / model_medium.glb"))
    lods = [f for f in LOD_FILES.values() if (folder / f).is_file()]
    add("LODs", "ok" if len(lods) > 1 else "info", ", ".join(lods))
    # ---- materials / textures
    if not doc.materials:
        out.append(Issue("warning", "no materials (vertex colours / white are used)"))
        add("Materials", "warn", "none (vertex colours / white are used)")
    else:
        add("Materials", "ok", ", ".join(f"{m.name or '?'}" + (" (unlit)" if m.unlit else "") for m in doc.materials))
    present, missing = [], []
    for m in doc.materials:
        for slot, ref in m.textures().items():
            (present if ref.ok else missing).append((m.name, slot, ref))
    if present or missing:
        add("Textures", "ok" if present else "warn",
            ", ".join(sorted({f"{slot}" for _, slot, _ in present})) + f" ({len(present)} loaded)")
    else:
        add("Textures", "info", "none (material colours only)")
    for mname, slot, ref in missing:
        out.append(Issue("warning", f"material {mname!r}: {slot} texture missing - {ref.error}"))
    add("Missing textures", "warn" if missing else "ok",
        "; ".join(f"{mn}.{sl}: {r.error}" for mn, sl, r in missing) if missing else "none")
    uvless = [n for n in meshes for p in doc.meshes[n.mesh].primitives
              if p.uvs is None and p.material is not None and p.material < len(doc.materials)
              and doc.materials[p.material].textures()]
    if uvless:
        out.append(Issue("warning", "textured primitives without TEXCOORD_0 - textures can't be applied"))
    if doc.extensions_required:
        add("Extensions", "info", ", ".join(doc.extensions_required))
    # ---- skeleton
    if not doc.skins and doc.animations:
        synth_skin(doc)
    extend_skin(doc, meta.get("bone_map"))
    if not doc.skins:
        out.append(Issue("error", "no skeleton / skin - the model must be rigged"))
        add("Skeleton", "fail", "no skin - the model must be rigged")
        return checks, out
    skin = doc.skins[0]
    names = [doc.nodes[j].name for j in skin.joints]
    canon = {canonical_name(n, meta.get("bone_map")) for n in names}
    add("Skeleton", "ok", f"{len(names)} joints" + (f", {len(doc.skins)} skins" if len(doc.skins) > 1 else ""))
    missing_b = [b for b in REQUIRED_BONES if b not in canon]
    if "Neck" in missing_b and "Head" in canon and "Chest" in canon:
        # stylised rigs (KayKit, chibi characters) often sit the head straight on the chest:
        # the rig driver then turns the head joint alone
        missing_b.remove("Neck")
        out.append(Issue("warning", "no Neck bone (head sits on the chest): head turns use the Head joint only"))
        add("Neck", "warn", "no Neck bone - head turns use the Head joint only")
    if missing_b:
        out.append(Issue("error", "missing required bones: " + ", ".join(missing_b) +
                         " (rename them or add a bone_map to metadata.json)"))
        add("Required bones", "fail", "missing " + ", ".join(missing_b) + " - rename or add a bone_map")
    else:
        add("Required bones", "ok", f"all {len(REQUIRED_BONES)} found")
    if kind == "quadruped" and not any(c and c.startswith("Tail") for c in canon):
        out.append(Issue("warning", "no tail bones (Tail0, Tail1 ...): the tail won't move"))
    for b in OPTIONAL_BONES[:3]:
        if b not in canon:
            out.append(Issue("info", f"optional bone {b} not found"))
    unweighted = 0
    rigid = 0
    for n in meshes:
        for p in doc.meshes[n.mesh].primitives:
            if p.joints is None or p.weights is None:
                if n.skin is None and _has_joint_ancestor(doc, n, set(skin.joints)):
                    rigid += len(p.positions)
                else:
                    unweighted += len(p.positions)
    if unweighted:
        out.append(Issue("error", f"{unweighted} vertices have no JOINTS_0 / WEIGHTS_0 (not skinned)"))
        add("Skin weights", "fail", f"{unweighted} vertices are neither skinned nor parented to a bone")
    else:
        bad = 0
        for n in meshes:
            for p in doc.meshes[n.mesh].primitives:
                if p.weights is not None:
                    bad += int((np.abs(p.weights.sum(axis=1) - 1.0) > 0.02).sum())
        add("Skin weights", "warn" if bad else "ok", (f"{bad} vertices with weights not summing to 1 (normalised)"
                                                      if bad else "4 influences, normalised")
            + (f"; {rigid} vertices on bone-parented rigid parts" if rigid else ""))
    # ---- bind pose geometry
    recs = _gather(doc) if meshes else []
    pts = np.concatenate([r["pos"] for r in recs]) if recs else None
    if pts is not None and len(pts):
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        h = hi[1] - lo[1]
        if h <= 0:
            out.append(Issue("error", "mesh has no height"))
            add("Scale", "fail", "mesh has no height")
        elif not (0.05 < h < 50):
            out.append(Issue("warning", f"height {h:.3f} units looks unusual (metres expected: ~0.2-2) - the loader "
                                        "normalises it anyway"))
            add("Scale", "warn", f"height {h:.3f} units (metres expected ~0.2-2; normalised anyway)")
        else:
            add("Scale", "ok", f"height {h:.2f} m")
        if abs(lo[1]) > 0.05 * max(h, 1e-6):
            out.append(Issue("warning", f"lowest vertex at Y = {lo[1]:.3f}: the feet should touch Y = 0 (ground)"))
            add("Ground alignment", "warn", f"lowest vertex at Y = {lo[1]:.3f} (feet should touch Y = 0; "
                                            "the loader shifts it)")
        else:
            add("Ground alignment", "ok", "feet on Y = 0")
        world = doc.world_matrices()
        idx = {canonical_name(doc.nodes[j].name, meta.get("bone_map")): j for j in skin.joints}
        head, pelvis = idx.get("Head"), idx.get("Pelvis")
        orient_ok = True
        if head is not None and pelvis is not None:
            hp, pp = world[head][:3, 3], world[pelvis][:3, 3]
            if hp[1] < pp[1] - 1e-6 and kind == "biped":
                out.append(Issue("error", "head is below the pelvis: wrong up axis (Y must be up)"))
                add("Orientation", "fail", "head below the pelvis - Y must be up")
                orient_ok = False
            if kind == "quadruped" and hp[2] < pp[2]:
                out.append(Issue("error", "head is behind the pelvis: the character must face +Z"))
                add("Orientation", "fail", "head behind the pelvis - the character must face +Z")
                orient_ok = False
            if kind == "biped":
                lf, rf = idx.get("LeftFoot"), idx.get("RightFoot")
                if lf is not None and rf is not None and world[lf][0, 3] < world[rf][0, 3]:
                    out.append(Issue("warning", "left foot is on -X: the character should face +Z (left = +X)"))
                    add("Orientation", "warn", "left foot on -X - should face +Z")
                    orient_ok = False
        if orient_ok:
            add("Orientation", "ok", "Y up, faces +Z")
    # ---- animations / morphs
    if not doc.animations:
        out.append(Issue("info", "no animation clips - procedural animation is used (clips are optional)"))
        add("Animations", "info", "none - procedural animation is used (clips are optional)")
    else:
        out.append(Issue("info", "clips: " + ", ".join(a.name for a in doc.animations)))
        desc = []
        for a in doc.animations:
            paths = sorted({c.path for c in a.channels})
            desc.append(f"{a.name} {a.duration:.2f}s [{'/'.join(p[0].upper() for p in paths)}]")
        add("Animations", "ok", ", ".join(desc))
    morphs = sorted({n for m in doc.meshes for n in (m.target_names or [])} |
                    {f"target{i}" for m in doc.meshes if not m.target_names
                     for p in m.primitives for i in range(len(p.targets))})
    if morphs:
        known = [m for m in morphs if "".join(ch for ch in m.lower() if ch.isalnum()) in ARKIT_MORPHS or
                 m in (meta.get("morph_map") or {}) or m in set((meta.get("morphs") or {}).values())]
        add("Morph targets", "ok", f"{len(morphs)}: " + ", ".join(morphs) +
            (f" ({len(known)} mapped to the face)" if known else " (none mapped - add \"morphs\" to metadata)"))
    else:
        add("Morph targets", "info", "none (decal / bone face)")
    if not meta.get("production"):
        out.append(Issue("info", "metadata says production=false (demonstration / placeholder asset)"))
        add("Production flag", "info", "production=false (demonstration / placeholder asset)")
    else:
        add("Production flag", "ok", "production=true")
    if not any(i.level == "error" for i in out):
        out.append(Issue("info", "OK - usable as a FocusBuddy character"))
    return checks, out


def _has_joint_ancestor(doc: gltf.Document, node, joints: set) -> bool:
    parents = doc.parents()
    i = doc.nodes.index(node)
    p = parents[i]
    while p is not None:
        if p in joints:
            return True
        p = parents[p]
    return False
