"""Synthetic DTM with a known embedded (rotated, raised) patch, a decoy with
twice the relief, and a copy hidden inside a hole of the search polygon."""
import math
import numpy as np
from scipy import ndimage as ndi

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from terrain_core import Grid


def patch_fn(u, v):
    """Asymmetric analytic terrain (m) in local metres (u east, v north)."""
    z = 0.06 * u - 0.03 * v                                   # overall slope
    z += 28 * np.exp(-(((u - 60) / 55) ** 2 + ((v - 40) / 30) ** 2))   # knoll
    z -= 24 * np.exp(-(((v - 0.35 * u + 30) / 14) ** 2)) * np.exp(-((u + 20) / 160) ** 2)  # valley
    z += 14 * np.exp(-(((u + 90) / 25) ** 2 + ((v + 60) / 50) ** 2))   # shoulder
    z -= 10 * np.exp(-(((u - 20) / 20) ** 2 + ((v + 90) / 18) ** 2))   # pit
    return z


def ref_rings(r0=95.0):
    """Irregular polygon with a hole, local metres around (0,0)."""
    phi = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    r = r0 * (1 + 0.28 * np.cos(2 * phi + 0.4) + 0.15 * np.sin(3 * phi))
    outer = np.column_stack([r * np.cos(phi), 0.8 * r * np.sin(phi)])[::-1]   # clockwise
    ph = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    hole = np.column_stack([18 + 16 * np.cos(ph), 10 + 10 * np.sin(ph)])      # ccw
    return [np.vstack([outer, outer[:1]]), np.vstack([hole, hole[:1]])]


def rot(p, th_deg):
    t = math.radians(th_deg)
    c, s = math.cos(t), math.sin(t)
    p = np.asarray(p, float)
    return np.column_stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c])


def make_dem(n=1000, cs=2.0, seed=3, ref_center=(300.0, 300.0),
             targets=((1400.0, 1450.0, 73.0, 100.0, 1.0),),
             decoys=((600.0, 1500.0, 200.0, -40.0, 2.0),),
             hidden=((1500.0, 400.0, 20.0, 50.0, 1.0),),
             R_in=140.0, R_out=175.0, relief_ref=None):
    """Returns dict with DEM array (north-up, NaN none), Grid and truth.
    Coordinates: origin (0,0) at lower-left, y north."""
    rng = np.random.default_rng(seed)
    bg = ndi.gaussian_filter(rng.standard_normal((n, n)), 9) * 150.0   # ~ +-15 m rolling
    bg += ndi.gaussian_filter(rng.standard_normal((n, n)), 3) * 25.0
    xs = (np.arange(n) + 0.5) * cs
    ys = (n - np.arange(n) - 0.5) * cs
    X, Y = np.meshgrid(xs, ys)
    z = bg.copy()
    # reference patch mean (for decoy amplitude scaling)
    rr = ref_rings()
    mref = float(patch_fn(*np.meshgrid(np.linspace(-80, 80, 20), np.linspace(-80, 80, 20))).mean())

    def embed(cx, cy, ang, dz, amp):
        nonlocal z
        d = np.hypot(X - cx, Y - cy)
        w = np.clip((R_out - d) / (R_out - R_in), 0, 1)
        w = w * w * (3 - 2 * w)
        k = w > 0
        q = rot(np.column_stack([(X - cx)[k], (Y - cy)[k]]), -ang)    # local coords
        f = patch_fn(q[:, 0], q[:, 1])
        f = mref + amp * (f - mref) + dz
        z[k] = w[k] * f + (1 - w[k]) * z[k]

    embed(*ref_center, 0.0, 0.0, 1.0)
    for t in targets:
        embed(*t)
    for t in decoys:
        embed(*t)
    for t in hidden:
        embed(*t)
    geo = Grid(0.0, n * cs, cs, n, n)
    truth = dict(ref_center=ref_center, targets=targets, decoys=decoys, hidden=hidden)
    return dict(z=z, geo=geo, truth=truth, ref_local=rr)


def world_rings(local_rings, cx, cy, ang):
    return [rot(r, ang) + np.array([cx, cy]) for r in local_rings]


def search_rings_for(extent, hole_center, hole_r=190.0, margin=30.0, exclude_ref=None):
    """Square search polygon with a circular hole (hidden copy) and a
    rectangular carve-out around the reference area."""
    xmin, ymin, xmax, ymax = extent
    outer = np.array([[xmin + margin, ymin + margin], [xmin + margin, ymax - margin],
                      [xmax - margin, ymax - margin], [xmax - margin, ymin + margin],
                      [xmin + margin, ymin + margin]], float)
    ph = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    hole = np.column_stack([hole_center[0] + hole_r * np.cos(ph), hole_center[1] + hole_r * np.sin(ph)])
    hole = np.vstack([hole[::-1], hole[::-1][:1]])
    rings = [outer, hole]
    if exclude_ref is not None:
        cx, cy, r = exclude_ref
        b = np.array([[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r],
                      [cx - r, cy - r]], float)
        rings.append(b)     # even-odd: a second hole
    return rings


def true_anchor(anchor_local, cx, cy, ang):
    a = rot(np.array([anchor_local]), ang)[0]
    return cx + a[0], cy + a[1]
