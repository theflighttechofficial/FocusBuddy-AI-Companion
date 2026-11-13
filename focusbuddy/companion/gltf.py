"""Minimal, dependency-free glTF 2.0 reader / writer (``.glb`` and ``.gltf``) built on NumPy.

Reads everything the character pipeline needs:

* meshes          POSITION, NORMAL, TANGENT, COLOR_0, TEXCOORD_0/1, JOINTS_0, WEIGHTS_0,
                  indices (or none), morph targets (POSITION / NORMAL deltas,
                  ``extras.targetNames``), material
* skins           joints + inverseBindMatrices
* nodes           hierarchy, names, TRS or matrix, mesh / skin references
* materials       glTF PBR metallic-roughness: base colour factor + texture, metallic /
                  roughness factors + texture, normal texture (+ scale), occlusion texture
                  (+ strength), emissive factor + texture (+ KHR_materials_emissive_strength),
                  alpha mode / cutoff, double-sided, KHR_materials_unlit, ``extras``.
                  Textures: embedded (GLB buffer view / data URI) or relative files (PNG,
                  JPEG, WebP via EXT_texture_webp) decoded with Qt; KHR_texture_transform.
* animations      translation / rotation / scale / weights channels, LINEAR / STEP / CUBICSPLINE
* extras          scene / asset ``extras`` (FocusBuddy metadata can live there too)

Buffers may be the GLB binary chunk, ``data:`` URIs or files next to a ``.gltf``.
Sparse accessors and KHR extensions that change geometry (Draco, meshopt) are not
supported - the validator reports them clearly.

The writer (``GLBWriter``) produces spec-compliant GLB files from NumPy arrays; it is
used by the demo-asset builder and by the tests.
"""

from __future__ import annotations

import base64
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class GLTFError(Exception):
    pass


# ----------------------------------------------------------------- math helpers
def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(m) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / (np.linalg.norm(q) or 1.0)


def slerp(a, b, t: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + (b - a) * t
        return r / (np.linalg.norm(r) or 1.0)
    th = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(th)
    return (a * math.sin((1 - t) * th) + b * math.sin(t * th)) / s


def trs_matrix(t=(0, 0, 0), r=(0, 0, 0, 1), s=(1, 1, 1)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(r) * np.asarray(s, dtype=float)[None, :]
    m[:3, 3] = t
    return m


# ------------------------------------------------------------------ data classes
@dataclass
class Primitive:
    positions: np.ndarray                     # (N, 3) float32
    indices: np.ndarray                       # (M, 3) int
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None       # (N, 3|4) float
    uvs: Optional[np.ndarray] = None          # (N, 2) TEXCOORD_0
    joints: Optional[np.ndarray] = None       # (N, 4) int
    weights: Optional[np.ndarray] = None      # (N, 4) float
    targets: list = field(default_factory=list)   # [{"POSITION": (N,3), "NORMAL": (N,3)}]
    material: Optional[int] = None
    tangents: Optional[np.ndarray] = None     # (N, 4) xyz + handedness
    uvs1: Optional[np.ndarray] = None         # (N, 2) TEXCOORD_1


@dataclass
class Mesh:
    name: str
    primitives: list
    target_names: list = field(default_factory=list)
    weights: list = field(default_factory=list)


@dataclass
class Node:
    name: str
    children: list
    translation: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    matrix: Optional[np.ndarray] = None
    mesh: Optional[int] = None
    skin: Optional[int] = None
    extras: dict = field(default_factory=dict)

    def local(self) -> np.ndarray:
        if self.matrix is not None:
            return self.matrix
        return trs_matrix(self.translation, self.rotation, self.scale)


@dataclass
class Skin:
    name: str
    joints: list                               # node indices
    inverse_bind: np.ndarray                   # (J, 4, 4)
    skeleton: Optional[int] = None


@dataclass
class TextureRef:
    """One texture slot of a material."""
    image: Optional[np.ndarray]                # (H, W, 4) uint8 RGBA, None when missing / undecodable
    texcoord: int = 0                          # TEXCOORD_n
    source: str = ""                           # "embedded", "data-uri" or the relative file name
    mime: str = ""
    error: str = ""                            # why it is missing
    transform: Optional[dict] = None           # KHR_texture_transform (offset, scale, rotation)
    wrap: tuple = (10497, 10497)               # REPEAT by default
    scale: float = 1.0                         # normalTexture.scale / occlusionTexture.strength

    @property
    def ok(self) -> bool:
        return self.image is not None


@dataclass
class Material:
    name: str
    base_color: tuple = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 1.0
    roughness: float = 1.0
    emissive: tuple = (0.0, 0.0, 0.0)
    texture: Optional[np.ndarray] = None       # (H, W, 4) float 0..1 base colour texture (legacy access)
    extras: dict = field(default_factory=dict)
    base_tex: Optional[TextureRef] = None
    normal_tex: Optional[TextureRef] = None
    mr_tex: Optional[TextureRef] = None        # G = roughness, B = metallic
    occlusion_tex: Optional[TextureRef] = None # R = occlusion
    emissive_tex: Optional[TextureRef] = None
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    unlit: bool = False
    emissive_strength: float = 1.0

    def textures(self) -> dict:
        return {k: v for k, v in (("baseColor", self.base_tex), ("normal", self.normal_tex),
                                  ("metallicRoughness", self.mr_tex), ("occlusion", self.occlusion_tex),
                                  ("emissive", self.emissive_tex)) if v is not None}


@dataclass
class Channel:
    node: int
    path: str                                  # translation | rotation | scale | weights
    times: np.ndarray
    values: np.ndarray
    interpolation: str = "LINEAR"


@dataclass
class Animation:
    name: str
    channels: list

    @property
    def duration(self) -> float:
        return max((float(c.times[-1]) for c in self.channels if len(c.times)), default=0.0)


@dataclass
class Document:
    nodes: list
    meshes: list
    skins: list
    materials: list
    animations: list
    scene_roots: list
    extras: dict = field(default_factory=dict)
    asset: dict = field(default_factory=dict)
    extensions_required: list = field(default_factory=list)
    source: str = ""

    def node_index(self, name: str) -> Optional[int]:
        for i, n in enumerate(self.nodes):
            if n.name == name:
                return i
        return None

    def parents(self) -> list:
        p = [None] * len(self.nodes)
        for i, n in enumerate(self.nodes):
            for c in n.children:
                p[c] = i
        return p

    def world_matrices(self) -> list:
        """World matrix of every node in the bind / rest pose."""
        out = [None] * len(self.nodes)
        parents = self.parents()

        def world(i):
            if out[i] is None:
                loc = self.nodes[i].local()
                out[i] = loc if parents[i] is None else world(parents[i]) @ loc
            return out[i]
        for i in range(len(self.nodes)):
            world(i)
        return out


# ------------------------------------------------------------------------ reader
def load(path) -> Document:
    path = Path(path)
    data = path.read_bytes()
    if data[:4] == b"glTF":
        gl, bin_chunk = _parse_glb(data)
    else:
        gl, bin_chunk = json.loads(data.decode("utf-8")), None
    return _build(gl, bin_chunk, path.parent, str(path))


def loads_glb(data: bytes) -> Document:
    gl, bin_chunk = _parse_glb(data)
    return _build(gl, bin_chunk, Path("."), "<bytes>")


def _parse_glb(data: bytes):
    if len(data) < 20:
        raise GLTFError("file too short for GLB")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise GLTFError("not a GLB file (bad magic)")
    if version != 2:
        raise GLTFError(f"unsupported glTF version {version}")
    off = 12
    gl, bin_chunk = None, None
    while off + 8 <= min(length, len(data)):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == CHUNK_JSON:
            gl = json.loads(chunk.decode("utf-8"))
        elif ctype == CHUNK_BIN and bin_chunk is None:
            bin_chunk = chunk
        off += 8 + clen
    if gl is None:
        raise GLTFError("GLB has no JSON chunk")
    return gl, bin_chunk


def _build(gl: dict, bin_chunk, base: Path, source: str) -> Document:
    req = gl.get("extensionsRequired", []) or []
    unsupported = [e for e in req if e in ("KHR_draco_mesh_compression", "EXT_meshopt_compression")]
    if unsupported:
        raise GLTFError(f"compressed geometry not supported: {', '.join(unsupported)} (export without compression)")
    buffers = []
    for b in gl.get("buffers", []):
        uri = b.get("uri")
        if uri is None:
            if bin_chunk is None:
                raise GLTFError("buffer without uri and no GLB binary chunk")
            buffers.append(bytes(bin_chunk))
        elif uri.startswith("data:"):
            buffers.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            buffers.append(_local(base, uri).read_bytes())
    views = gl.get("bufferViews", [])
    accessors = gl.get("accessors", [])

    def acc(i):
        if i is None:
            return None
        a = accessors[i]
        if "sparse" in a:
            raise GLTFError("sparse accessors are not supported")
        dt = _COMP[a["componentType"]]
        nc = _NCOMP[a["type"]]
        count = a["count"]
        if "bufferView" not in a:
            return np.zeros((count, nc) if nc > 1 else count, dtype=dt)
        v = views[a["bufferView"]]
        buf = buffers[v["buffer"]]
        off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        item = np.dtype(dt).itemsize * nc
        stride = v.get("byteStride", 0) or item
        if stride == item:
            arr = np.frombuffer(buf, dtype=dt, count=count * nc, offset=off).copy()
        else:
            raw = np.frombuffer(buf, dtype=np.uint8, count=stride * (count - 1) + item, offset=off)
            rows = np.lib.stride_tricks.as_strided(raw, shape=(count, item), strides=(stride, 1))
            arr = np.frombuffer(rows.copy().tobytes(), dtype=dt)
        if a.get("normalized") and dt != np.float32:
            arr = arr.astype(np.float32) / float(np.iinfo(dt).max)
        return arr.reshape(count, nc) if nc > 1 else arr

    textures = gl.get("textures", [])
    images = gl.get("images", [])
    samplers = gl.get("samplers", [])
    decoded: dict = {}

    def image(src):
        """Decoded RGBA8 image (cached per image index) + (source, mime, error)."""
        if src in decoded:
            return decoded[src]
        im = images[src] if src is not None and 0 <= src < len(images) else {}
        source, mime, arr, err = "", im.get("mimeType", ""), None, ""
        try:
            if not im:
                err = "texture has no image"
            elif "bufferView" in im:
                v = views[im["bufferView"]]
                raw = buffers[v["buffer"]][v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]]
                source = "embedded"
                arr = _decode_image(raw)
            elif im.get("uri", "").startswith("data:"):
                source = "data-uri"
                arr = _decode_image(base64.b64decode(im["uri"].split(",", 1)[1]))
            else:
                source = im.get("uri", "")
                path = _local(base, source)
                if not path.is_file():
                    err = f"missing file {source}"
                else:
                    arr = _decode_image(path.read_bytes())
            if arr is None and not err:
                err = f"cannot decode {source or 'image'} ({mime or 'unknown type'})"
        except Exception as e:  # noqa: BLE001 - a missing / undecodable texture falls back to the factors
            err = str(e)
            arr = None
        decoded[src] = (arr, source, mime, err)
        return decoded[src]

    def texref(info: Optional[dict], scale_key: str = "") -> Optional[TextureRef]:
        if not info or "index" not in info:
            return None
        ti = info["index"]
        if not (0 <= ti < len(textures)):
            return TextureRef(None, error=f"texture index {ti} out of range")
        tex = textures[ti]
        src = tex.get("source")
        ext = tex.get("extensions", {}) or {}
        for key in ("EXT_texture_webp", "KHR_texture_basisu", "MSFT_texture_dds"):
            if key in ext and "source" in ext[key]:
                if key == "EXT_texture_webp" or src is None:
                    src = ext[key]["source"]
        arr, source, mime, err = image(src)
        smp = samplers[tex["sampler"]] if "sampler" in tex and tex["sampler"] < len(samplers) else {}
        tr = (info.get("extensions", {}) or {}).get("KHR_texture_transform")
        return TextureRef(arr, int((tr or {}).get("texCoord", info.get("texCoord", 0))), source, mime, err, tr,
                          (smp.get("wrapS", 10497), smp.get("wrapT", 10497)),
                          float(info.get(scale_key, 1.0)) if scale_key else 1.0)

    materials = []
    for m in gl.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        ext = m.get("extensions", {}) or {}
        base_tex = texref(pbr.get("baseColorTexture"))
        legacy = base_tex.image.astype(np.float32) / 255.0 if base_tex is not None and base_tex.ok else None
        materials.append(Material(
            m.get("name", ""), tuple(pbr.get("baseColorFactor", (1, 1, 1, 1))),
            float(pbr.get("metallicFactor", 1.0)), float(pbr.get("roughnessFactor", 1.0)),
            tuple(m.get("emissiveFactor", (0, 0, 0))), legacy, m.get("extras", {}) or {},
            base_tex, texref(m.get("normalTexture"), "scale"), texref(pbr.get("metallicRoughnessTexture")),
            texref(m.get("occlusionTexture"), "strength"), texref(m.get("emissiveTexture")),
            m.get("alphaMode", "OPAQUE"), float(m.get("alphaCutoff", 0.5)), bool(m.get("doubleSided", False)),
            "KHR_materials_unlit" in ext,
            float((ext.get("KHR_materials_emissive_strength") or {}).get("emissiveStrength", 1.0))))
    meshes = []
    for m in gl.get("meshes", []):
        prims = []
        for p in m.get("primitives", []):
            if p.get("mode", 4) != 4:
                continue                         # only triangle lists
            at = p.get("attributes", {})
            pos = acc(at.get("POSITION"))
            if pos is None:
                continue
            idx = acc(p.get("indices"))
            if idx is None:
                idx = np.arange(len(pos))
            idx = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
            col = acc(at.get("COLOR_0"))
            if col is not None and col.dtype != np.float32:
                col = col.astype(np.float32)
            targets = []
            for t in p.get("targets", []):
                targets.append({k: acc(v) for k, v in t.items() if k in ("POSITION", "NORMAL")})
            prims.append(Primitive(pos.astype(np.float32), idx, acc(at.get("NORMAL")), col,
                                   _f32(acc(at.get("TEXCOORD_0"))), acc(at.get("JOINTS_0")),
                                   _f32(acc(at.get("WEIGHTS_0"))), targets, p.get("material"),
                                   _f32(acc(at.get("TANGENT"))), _f32(acc(at.get("TEXCOORD_1")))))
        extras = m.get("extras", {}) or {}
        meshes.append(Mesh(m.get("name", ""), prims, list(extras.get("targetNames", [])), list(m.get("weights", []))))
    nodes = []
    for n in gl.get("nodes", []):
        mat = None
        if "matrix" in n:
            mat = np.array(n["matrix"], dtype=float).reshape(4, 4).T     # glTF is column-major
        nodes.append(Node(n.get("name", ""), list(n.get("children", [])),
                          np.array(n.get("translation", (0, 0, 0)), dtype=float),
                          np.array(n.get("rotation", (0, 0, 0, 1)), dtype=float),
                          np.array(n.get("scale", (1, 1, 1)), dtype=float), mat, n.get("mesh"), n.get("skin"),
                          n.get("extras", {}) or {}))
    skins = []
    for s in gl.get("skins", []):
        ib = acc(s.get("inverseBindMatrices"))
        joints = list(s.get("joints", []))
        if ib is None:
            ibm = np.repeat(np.eye(4)[None], len(joints), axis=0)
        else:
            ibm = np.asarray(ib, dtype=float).reshape(-1, 4, 4).transpose(0, 2, 1)
        skins.append(Skin(s.get("name", ""), joints, ibm, s.get("skeleton")))
    anims = []
    for a in gl.get("animations", []):
        chans = []
        samplers = a.get("samplers", [])
        for c in a.get("channels", []):
            tgt = c.get("target", {})
            if "node" not in tgt:
                continue
            s = samplers[c["sampler"]]
            times = np.asarray(acc(s["input"]), dtype=float).reshape(-1)
            vals = np.asarray(acc(s["output"]), dtype=float)
            if vals.ndim == 1:
                vals = vals.reshape(len(times) * (3 if s.get("interpolation") == "CUBICSPLINE" else 1), -1)
            chans.append(Channel(tgt["node"], tgt["path"], times, vals, s.get("interpolation", "LINEAR")))
        anims.append(Animation(a.get("name", f"clip{len(anims)}"), chans))
    scene = gl.get("scenes", [{}])[gl.get("scene", 0)] if gl.get("scenes") else {}
    roots = list(scene.get("nodes", [])) or [i for i, p in enumerate(_parents(nodes)) if p is None]
    extras = dict(gl.get("asset", {}).get("extras", {}) or {})
    extras.update(scene.get("extras", {}) or {})
    return Document(nodes, meshes, skins, materials, anims, roots, extras, gl.get("asset", {}), req, source)


def _f32(a):
    return None if a is None else np.asarray(a, dtype=np.float32)


def _local(base: Path, uri: str) -> Path:
    """External buffer / image next to the .gltf - never outside the asset folder."""
    from urllib.parse import unquote
    p = (base / unquote(uri)).resolve()
    root = base.resolve()
    if root != p and root not in p.parents:
        raise GLTFError(f"external file outside the asset folder: {uri}")
    return p


def _parents(nodes) -> list:
    p = [None] * len(nodes)
    for i, n in enumerate(nodes):
        for c in n.children:
            if 0 <= c < len(nodes):
                p[c] = i
    return p


def _decode_image(raw: bytes) -> Optional[np.ndarray]:
    try:
        from PySide6.QtGui import QImage
    except Exception:  # noqa: BLE001
        return None
    img = QImage()
    if not img.loadFromData(raw):
        return None
    img = img.convertToFormat(QImage.Format_RGBA8888)
    w, h = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8, count=img.bytesPerLine() * h)
    return arr.reshape(h, img.bytesPerLine())[:, :w * 4].reshape(h, w, 4).copy()


# --------------------------------------------------------------------- sampling
def sample_channel(c: Channel, t: float) -> np.ndarray:
    times, vals = c.times, c.values
    if len(times) == 0:
        return vals[0] if len(vals) else np.zeros(4)
    cubic = c.interpolation == "CUBICSPLINE"
    if cubic:
        vals = vals.reshape(len(times), 3, -1)
    if t <= times[0]:
        return vals[0, 1] if cubic else vals[0]
    if t >= times[-1]:
        return vals[-1, 1] if cubic else vals[-1]
    i = int(np.searchsorted(times, t) - 1)
    t0, t1 = times[i], times[i + 1]
    u = (t - t0) / max(1e-9, t1 - t0)
    if c.interpolation == "STEP":
        return vals[i]
    if cubic:
        dt = t1 - t0
        p0, m0 = vals[i, 1], vals[i, 2] * dt
        p1, m1 = vals[i + 1, 1], vals[i + 1, 0] * dt
        u2, u3 = u * u, u * u * u
        r = (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1
        return r / (np.linalg.norm(r) or 1.0) if c.path == "rotation" else r
    if c.path == "rotation":
        return slerp(vals[i], vals[i + 1], u)
    return vals[i] + (vals[i + 1] - vals[i]) * u


# ------------------------------------------------------------------------ writer
class GLBWriter:
    """Build a GLB from NumPy arrays: nodes, one skinned mesh per call, materials, animations."""

    def __init__(self, generator: str = "FocusBuddy"):
        self.gl: dict = {"asset": {"version": "2.0", "generator": generator}, "scene": 0,
                         "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "accessors": [], "bufferViews": [],
                         "buffers": [], "materials": []}
        self.bin = bytearray()

    def _view(self, data: bytes, target: Optional[int] = None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            v["target"] = target
        self.bin.extend(data)
        self.gl["bufferViews"].append(v)
        return len(self.gl["bufferViews"]) - 1

    def accessor(self, arr: np.ndarray, kind: str, comp: int = 5126, target: Optional[int] = None,
                 minmax: bool = False, normalized: bool = False) -> int:
        dt = _COMP[comp]
        a = np.ascontiguousarray(arr, dtype=dt)
        if kind.startswith("MAT"):
            n = int(math.isqrt(_NCOMP[kind]))
            a = np.ascontiguousarray(a.reshape(-1, n, n).transpose(0, 2, 1), dtype=dt)
        count = a.shape[0]
        view = self._view(a.tobytes(), target)
        acc = {"bufferView": view, "componentType": comp, "count": int(count), "type": kind}
        if normalized:
            acc["normalized"] = True
        if minmax:
            flat = a.reshape(count, -1)
            acc["min"] = [float(v) for v in flat.min(axis=0)]
            acc["max"] = [float(v) for v in flat.max(axis=0)]
        self.gl["accessors"].append(acc)
        return len(self.gl["accessors"]) - 1

    def add_texture(self, rgba: np.ndarray, mime: str = "image/png", quality: int = 92) -> int:
        """Embed an (H, W, 3|4) uint8 image as a PNG / JPEG (``quality``) texture; returns the texture index."""
        from PySide6.QtCore import QBuffer, QByteArray, QIODevice
        from PySide6.QtGui import QImage
        a = np.ascontiguousarray(rgba, dtype=np.uint8)
        if a.shape[2] == 3:
            a = np.ascontiguousarray(np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], axis=2))
        h, w = a.shape[:2]
        img = QImage(a.tobytes(), w, h, w * 4, QImage.Format_RGBA8888).copy()
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.WriteOnly)
        if mime == "image/jpeg":
            img.save(buf, "JPG", quality)
        else:
            img.save(buf, "PNG")
        buf.close()
        view = self._view(bytes(ba.data()))
        self.gl.setdefault("images", []).append({"bufferView": view, "mimeType": mime})
        self.gl.setdefault("samplers", [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}])
        self.gl.setdefault("textures", []).append({"sampler": 0, "source": len(self.gl["images"]) - 1})
        return len(self.gl["textures"]) - 1

    def add_material(self, name: str, color=(1, 1, 1, 1), metallic: float = 0.0, roughness: float = 0.8,
                     emissive=(0, 0, 0), extras: Optional[dict] = None, base_tex: Optional[int] = None,
                     normal_tex: Optional[int] = None, normal_scale: float = 1.0, mr_tex: Optional[int] = None,
                     occlusion_tex: Optional[int] = None, emissive_tex: Optional[int] = None,
                     double_sided: bool = False) -> int:
        m = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": [float(c) for c in color],
                                                    "metallicFactor": float(metallic),
                                                    "roughnessFactor": float(roughness)},
             "emissiveFactor": [float(c) for c in emissive]}
        if base_tex is not None:
            m["pbrMetallicRoughness"]["baseColorTexture"] = {"index": base_tex}
        if mr_tex is not None:
            m["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": mr_tex}
        if normal_tex is not None:
            m["normalTexture"] = {"index": normal_tex, "scale": float(normal_scale)}
        if occlusion_tex is not None:
            m["occlusionTexture"] = {"index": occlusion_tex}
        if emissive_tex is not None:
            m["emissiveTexture"] = {"index": emissive_tex}
        if double_sided:
            m["doubleSided"] = True
        if extras:
            m["extras"] = extras
        self.gl["materials"].append(m)
        return len(self.gl["materials"]) - 1

    def add_node(self, name: str, translation=(0, 0, 0), rotation=(0, 0, 0, 1), parent: Optional[int] = None,
                 root: bool = False, **extra) -> int:
        n = {"name": name}
        if any(abs(v) > 0 for v in translation):
            n["translation"] = [float(v) for v in translation]
        if tuple(rotation) != (0, 0, 0, 1):
            n["rotation"] = [float(v) for v in rotation]
        n.update(extra)
        self.gl["nodes"].append(n)
        i = len(self.gl["nodes"]) - 1
        if parent is not None:
            self.gl["nodes"][parent].setdefault("children", []).append(i)
        if root:
            self.gl["scenes"][0]["nodes"].append(i)
        return i

    def add_mesh(self, name: str, positions, indices, normals=None, colors=None, joints=None, weights=None,
                 material: Optional[int] = None, targets: Optional[list] = None, target_names=None,
                 uvs=None, tangents=None) -> int:
        at = {"POSITION": self.accessor(positions, "VEC3", target=34962, minmax=True)}
        if normals is not None:
            at["NORMAL"] = self.accessor(normals, "VEC3", target=34962)
        if uvs is not None:
            at["TEXCOORD_0"] = self.accessor(uvs, "VEC2", target=34962)
        if tangents is not None:
            at["TANGENT"] = self.accessor(tangents, "VEC4", target=34962)
        if colors is not None:
            at["COLOR_0"] = self.accessor(colors, "VEC4" if np.asarray(colors).shape[1] == 4 else "VEC3",
                                          target=34962)
        if joints is not None:
            at["JOINTS_0"] = self.accessor(joints, "VEC4", comp=5123, target=34962)
            at["WEIGHTS_0"] = self.accessor(weights, "VEC4", target=34962)
        prim = {"attributes": at, "indices": self.accessor(np.asarray(indices).reshape(-1), "SCALAR", comp=5125,
                                                           target=34963), "mode": 4}
        if material is not None:
            prim["material"] = material
        mesh = {"name": name, "primitives": [prim]}
        if targets:
            prim["targets"] = []
            for t in targets:
                d = {"POSITION": self.accessor(t["POSITION"], "VEC3", minmax=True)}
                if t.get("NORMAL") is not None:
                    d["NORMAL"] = self.accessor(t["NORMAL"], "VEC3")
                prim["targets"].append(d)
            mesh["weights"] = [0.0] * len(targets)
            mesh["extras"] = {"targetNames": list(target_names or [])}
        self.gl["meshes"].append(mesh)
        return len(self.gl["meshes"]) - 1

    def add_skin(self, joints: list, inverse_bind: np.ndarray, skeleton: Optional[int] = None) -> int:
        s = {"joints": list(joints), "inverseBindMatrices": self.accessor(np.asarray(inverse_bind), "MAT4")}
        if skeleton is not None:
            s["skeleton"] = skeleton
        self.gl.setdefault("skins", []).append(s)
        return len(self.gl["skins"]) - 1

    def add_animation(self, name: str, channels: list) -> int:
        """channels: [(node, path, times (K,), values (K, n))]"""
        anim = {"name": name, "channels": [], "samplers": []}
        for node, path, times, values in channels:
            si = len(anim["samplers"])
            vals = np.asarray(values, dtype=float)
            if path == "weights":                      # morph weights: one scalar per target per key
                out = self.accessor(vals.reshape(-1, 1), "SCALAR")
            else:
                out = self.accessor(vals, {3: "VEC3", 4: "VEC4"}[vals.shape[1]])
            anim["samplers"].append({"input": self.accessor(np.asarray(times, dtype=float).reshape(-1, 1), "SCALAR",
                                                            minmax=True),
                                     "output": out, "interpolation": "LINEAR"})
            anim["channels"].append({"sampler": si, "target": {"node": node, "path": path}})
        self.gl.setdefault("animations", []).append(anim)
        return len(self.gl["animations"]) - 1

    def to_bytes(self, extras: Optional[dict] = None) -> bytes:
        gl = dict(self.gl)
        while len(self.bin) % 4:
            self.bin.append(0)
        gl["buffers"] = [{"byteLength": len(self.bin)}]
        if extras:
            gl["asset"] = dict(gl["asset"], extras=extras)
        if not gl["materials"]:
            del gl["materials"]
        js = json.dumps(gl, separators=(",", ":")).encode("utf-8")
        while len(js) % 4:
            js += b" "
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        out = struct.pack("<III", GLB_MAGIC, 2, total)
        out += struct.pack("<II", len(js), CHUNK_JSON) + js
        out += struct.pack("<II", len(self.bin), CHUNK_BIN) + bytes(self.bin)
        return out

    def save(self, path, extras: Optional[dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(self.to_bytes(extras))
