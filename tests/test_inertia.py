"""Pose inertialization (companion/inertia.py): continuous motion passes, jumps become smooth."""

import math

import numpy as np

from focusbuddy.companion.inertia import PoseInertializer, mat_to_quat, quat_to_mat

# a small chain: root -> a -> b -> c (bones of length 10 along +x)
PARENT = [-1, 0, 1, 2]
BONE = np.array([10.0, 0.0, 0.0])


def rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def fk(angles):
    """World rotations / positions from local z-rotations per joint."""
    D, T = np.zeros((4, 3, 3)), np.zeros((4, 3))
    for i, ang in enumerate(angles):
        L = rz(ang)
        if PARENT[i] < 0:
            D[i], T[i] = L, 0.0
        else:
            p = PARENT[i]
            D[i] = D[p] @ L
            T[i] = T[p] + D[p] @ BONE
    return D, T


def test_quaternion_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quat_to_mat(q[None])[0]
        q2 = mat_to_quat(R[None])[0]
        assert np.allclose(np.abs(q @ q2), 1.0, atol=1e-9)


def test_smooth_motion_passes_through_exactly():
    f = PoseInertializer(PARENT, scale=30.0)
    for k in range(240):
        t = k / 60
        ang = [0.0, 40 * math.sin(t * 3), 30 * math.sin(t * 5), 20 * math.cos(t * 4)]
        D, T = fk(ang)
        D2, T2 = f.apply(D, T, 1 / 60)
        assert np.allclose(D2, D) and np.allclose(T2, T)
    assert f.events == 0


def test_a_jump_is_absorbed_and_fades():
    f = PoseInertializer(PARENT, scale=30.0)
    outs = []
    for k in range(120):
        ang = [0.0, 0.0, 0.0 if k < 30 else 80.0, 0.0]     # joint 2 snaps 80 degrees at frame 30
        D, T = fk(ang)
        D2, T2 = f.apply(D, T, 1 / 60)
        outs.append((T2.copy(), T.copy()))
    assert f.events >= 1
    tips = np.array([o[0][3] for o in outs])
    step = np.linalg.norm(np.diff(tips, axis=0), axis=1)
    assert step.max() < 3.0                                    # the tip never teleports (unfiltered: ~13)
    assert np.allclose(outs[-1][0], outs[-1][1], atol=1e-3)    # ... and arrives where the pose wants


def test_bones_stay_rigid_while_settling():
    f = PoseInertializer(PARENT, scale=30.0)
    for k in range(60):
        D, T = fk([0.0, 0.0 if k < 20 else 90.0, 0.0, 0.0])
        D2, T2 = f.apply(D, T, 1 / 60)
        for i in range(1, 4):
            assert abs(np.linalg.norm(T2[i] - T2[PARENT[i]]) - 10.0) < 1e-6


def test_excluded_joints_and_scale():
    f = PoseInertializer(PARENT, exclude=[2], scale=30.0)
    for k in range(40):
        D, T = fk([0.0, 0.0, 0.0 if k < 20 else 80.0, 0.0])
        D[3] *= 0.8                                         # a clenched fist: scaled bone
        D2, T2 = f.apply(D, T, 1 / 60)
        assert np.allclose(D2, D) and np.allclose(T2, T)    # the jumping joint is excluded: untouched
        assert abs(np.cbrt(np.linalg.det(D2[3])) - 0.8) < 1e-9


def test_long_pause_starts_fresh():
    f = PoseInertializer(PARENT, scale=30.0)
    D, T = fk([0, 0, 0, 0])
    f.apply(D, T, 1 / 60)
    f.apply(D, T, 1 / 60)
    D, T = fk([0, 90, 0, 0])
    D2, T2 = f.apply(D, T, 2.0)                             # hidden for two seconds: no blend
    assert np.allclose(T2, T)
