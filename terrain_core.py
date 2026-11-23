"""
terrain_core.py - numerical engine for the "Find Similar Terrain" toolbox.

Pure NumPy / SciPy (no arcpy import) so it can be unit-tested and benchmarked
outside ArcGIS Pro.  The .pyt file only does GIS I/O and calls
:func:`find_similar_terrain`.

Score (per candidate placement + rotation)
------------------------------------------
    d    = candidate elevation - reference elevation   (cell by cell)
    RMSE = sqrt(mean((d - mean(d))**2))                 [meters]
    offset = mean(d)                                    [meters]
A constant vertical shift scores ~0; relief amplitude and overall slope are
NOT normalised away.

Rotation convention
-------------------
Angle theta is in degrees, counter-clockwise when viewed from above
(+Z towards the viewer, x = east, y = north, i.e. the usual mathematical
convention), range [0, 360).  The reference footprint *and* its terrain are
rotated together about the reference anchor (centroid of the reference
cells) by +theta and then translated so the anchor sits at the match
position.  Pure rotation + translation: horizontal scale and handedness
(no mirroring) are preserved.

Search strategy
---------------
1. DEM -> comparison grid (exact area-weighted aggregation, NoData aware).
2. Coarse scan: for every angle in a coarse angular grid the masked,
   mean-removed RMSE is evaluated at *every* grid placement with FFT
   correlations (4 masked cross-correlations, tiled with halo overlap).
   Exact identity used:
       sum(d^2) = corr(S^2,m) - 2 corr(S,T*m) + sum(T^2 m)
       sum(d)   = corr(S,m) - sum(T m)
   Placements whose template touches NoData / outside the search area are
   rejected via corr(valid,m) < n.
3. Shortlist of geographically distinct seeds from the best-over-angle map.
4. Continuous local refinement of (x, y, angle) for every seed by a
   shrinking-stencil pattern search that scores with bilinear sampling
   (vectorised over the reference cells).  The step size is tied to the
   footprint: d(angle) = d(position) / R_max so a rotation step never moves
   the outer edge of the polygon by more than the position step.
5. Optional refinement at native resolution; final ranking uses the final
   scoring resolution.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi
from scipy import sparse

WORKERS = max(1, os.cpu_count() or 1)
MAX_SEARCH_CELLS = 60e6      # refuse search grids larger than this
MAX_FFT_TILE = 2048          # max FFT edge length (cells) incl. halo


# --------------------------------------------------------------------------
# Grids and rasterisation
# --------------------------------------------------------------------------
@dataclass
class Grid:
    """North-up grid.  Cell (r, c) centre = (x0+(c+.5)cs, ytop-(r+.5)cs)."""
    x0: float
    ytop: float
    cs: float
    nrows: int
    ncols: int

    @property
    def shape(self):
        return (self.nrows, self.ncols)

    @property
    def xmax(self):
        return self.x0 + self.ncols * self.cs

    @property
    def ymin(self):
        return self.ytop - self.nrows * self.cs

    def centers(self):
        xs = self.x0 + (np.arange(self.ncols) + 0.5) * self.cs
        ys = self.ytop - (np.arange(self.nrows) + 0.5) * self.cs
        return xs, ys


def ring_bbox(rings):
    pts = np.vstack([np.asarray(r, float)[:, :2] for r in rings])
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def snap_grid(bbox, cs, origin_x, origin_ytop, pad_cells=1):
    """Grid of cell size cs aligned to the lattice (origin_x, origin_ytop)
    that covers bbox (+pad_cells)."""
    xmin, ymin, xmax, ymax = bbox
    xmin -= pad_cells * cs
    ymin -= pad_cells * cs
    xmax += pad_cells * cs
    ymax += pad_cells * cs
    x0 = origin_x + math.floor((xmin - origin_x) / cs + 1e-9) * cs
    ytop = origin_ytop - math.floor((origin_ytop - ymax) / cs + 1e-9) * cs
    ncols = int(math.ceil((xmax - x0) / cs - 1e-9))
    nrows = int(math.ceil((ytop - ymin) / cs - 1e-9))
    return Grid(x0, ytop, cs, max(nrows, 1), max(ncols, 1))


def rasterize_rings(rings, g: Grid):
    """Even-odd polygon fill on cell centres (holes work automatically).
    `rings` is a list of (N,2) arrays: outer rings and holes together."""
    rows_all, x_all = [], []
    for ring in rings:
        r = np.asarray(ring, float)[:, :2]
        if len(r) < 3:
            continue
        if not np.allclose(r[0], r[-1]):
            r = np.vstack([r, r[:1]])
        x1, y1, x2, y2 = r[:-1, 0], r[:-1, 1], r[1:, 0], r[1:, 1]
        ymin = np.minimum(y1, y2)
        ymax = np.maximum(y1, y2)
        k = ymax > ymin
        x1, y1, x2, y2, ymin, ymax = (a[k] for a in (x1, y1, x2, y2, ymin, ymax))
        if len(x1) == 0:
            continue
        rlo = np.floor((g.ytop - ymax) / g.cs - 0.5).astype(np.int64) + 1
        rhi = np.floor((g.ytop - ymin) / g.cs - 0.5).astype(np.int64)
        rlo = np.maximum(rlo, 0)
        rhi = np.minimum(rhi, g.nrows - 1)
        cnt = np.maximum(rhi - rlo + 1, 0)
        tot = int(cnt.sum())
        if tot == 0:
            continue
        e = np.repeat(np.arange(len(cnt)), cnt)
        start = np.cumsum(cnt) - cnt
        rows = rlo[e] + (np.arange(tot) - start[e])
        yc = g.ytop - (rows + 0.5) * g.cs
        xi = x1[e] + (yc - y1[e]) * (x2[e] - x1[e]) / (y2[e] - y1[e])
        rows_all.append(rows)
        x_all.append(xi)
    out = np.zeros(g.shape, bool)
    if not rows_all:
        return out
    R = np.concatenate(rows_all)
    X = np.concatenate(x_all)
    order = np.lexsort((X, R))
    R, X = R[order], X[order]
    first = np.r_[True, R[1:] != R[:-1]]
    gstart = np.flatnonzero(first)
    gid = np.cumsum(first) - 1
    rank = np.arange(len(R)) - gstart[gid]
    ev = np.flatnonzero(rank % 2 == 0)
    ev = ev[ev + 1 < len(R)]
    ev = ev[R[ev + 1] == R[ev]]
    c0 = np.clip(np.ceil((X[ev] - g.x0) / g.cs - 0.5), 0, g.ncols).astype(np.int64)
    c1 = np.clip(np.ceil((X[ev + 1] - g.x0) / g.cs - 0.5), 0, g.ncols).astype(np.int64)
    diff = np.zeros((g.nrows, g.ncols + 1), np.int32)
    np.add.at(diff, (R[ev], c0), 1)
    np.add.at(diff, (R[ev], c1), -1)
    return np.cumsum(diff, axis=1)[:, :-1] > 0


# --------------------------------------------------------------------------
# DEM access and aggregation
# --------------------------------------------------------------------------
class ArrayDEM:
    """In-memory DEM with the same interface the ArcPy reader implements.
    arr uses NaN for NoData.  geo is the native Grid."""

    def __init__(self, arr, geo: Grid):
        self.arr = arr
        self.geo = geo

    def window(self, xmin, ymin, xmax, ymax):
        g = self.geo
        c0 = max(int(math.floor((xmin - g.x0) / g.cs + 1e-9)), 0)
        c1 = min(int(math.ceil((xmax - g.x0) / g.cs - 1e-9)), g.ncols)
        r0 = max(int(math.floor((g.ytop - ymax) / g.cs + 1e-9)), 0)
        r1 = min(int(math.ceil((g.ytop - ymin) / g.cs - 1e-9)), g.nrows)
        if c1 <= c0 or r1 <= r0:
            return None, None
        return (self.arr[r0:r1, c0:c1],
                Grid(g.x0 + c0 * g.cs, g.ytop - r0 * g.cs, g.cs, r1 - r0, c1 - c0))


def _axis_weights(n_nat, nat_origin, nat_cs, n_out, out_origin, out_cs):
    """Sparse (n_out x n_nat) matrix of overlap lengths (native-cell units)
    between output cells and native cells along one axis."""
    a = out_origin + np.arange(n_out) * out_cs
    b = a + out_cs
    ilo = np.clip(np.floor((a - nat_origin) / nat_cs + 1e-9).astype(np.int64), 0, n_nat)
    ihi = np.clip(np.ceil((b - nat_origin) / nat_cs - 1e-9).astype(np.int64), 0, n_nat)
    cnt = np.maximum(ihi - ilo, 0)
    tot = int(cnt.sum())
    if tot == 0:
        return sparse.csr_matrix((n_out, n_nat))
    j = np.repeat(np.arange(n_out), cnt)
    start = np.cumsum(cnt) - cnt
    idx = ilo[j] + (np.arange(tot) - start[j])
    lo = np.maximum(a[j], nat_origin + idx * nat_cs)
    hi = np.minimum(b[j], nat_origin + (idx + 1) * nat_cs)
    w = np.maximum(hi - lo, 0) / nat_cs
    k = w > 1e-12
    return sparse.csr_matrix((w[k], (j[k], idx[k])), shape=(n_out, n_nat))


def aggregate(arr, geo: Grid, grid: Grid, min_cov=0.5):
    """Area-weighted mean of the valid native cells inside each output cell.
    Output cell is valid only if >= min_cov of its area is covered by valid
    native cells (NoData is never averaged in as zero)."""
    Wy = _axis_weights(geo.nrows, 0.0, geo.cs, grid.nrows, geo.ytop - grid.ytop, grid.cs)
    Wx = _axis_weights(geo.ncols, geo.x0, geo.cs, grid.ncols, grid.x0, grid.cs)
    valid = np.isfinite(arr)
    a0 = np.where(valid, arr, 0.0).astype(np.float64)
    num = (Wx @ (Wy @ a0).T).T
    den = (Wx @ (Wy @ valid.astype(np.float64)).T).T
    cov = den / (grid.cs / geo.cs) ** 2
    out = np.full(grid.shape, np.nan)
    ok = cov >= min_cov
    out[ok] = num[ok] / den[ok]
    return out


def read_aggregated(dem, grid: Grid, min_cov=0.5):
    """dem.window()-based aggregation to `grid`, in row strips so that huge
    DEMs are never fully resident in memory."""
    ncs = dem.geo.cs
    strip = max(8, int(4096 * ncs / grid.cs))
    out = np.full(grid.shape, np.nan)
    for r0 in range(0, grid.nrows, strip):
        nb = min(strip, grid.nrows - r0)
        sub = Grid(grid.x0, grid.ytop - r0 * grid.cs, grid.cs, nb, grid.ncols)
        arr, g = dem.window(sub.x0 - ncs, sub.ymin - ncs, sub.xmax + ncs, sub.ytop + ncs)
        if arr is None:
            continue
        out[r0:r0 + nb] = aggregate(arr, g, sub, min_cov)
    return out


# --------------------------------------------------------------------------
# Reference template + FFT scan
# --------------------------------------------------------------------------
class RefTemplate:
    """Reference terrain on the comparison grid (z with NaN, boolean mask)."""

    def __init__(self, z, mask, grid: Grid):
        self.grid = grid
        self.cs = grid.cs
        self.mask = mask & np.isfinite(z)
        rr, cc = np.nonzero(self.mask)
        self.n = len(rr)
        self.ar = rr.mean()
        self.ac = cc.mean()
        dist = np.hypot(rr - self.ar, cc - self.ac)
        self.R = float(dist.max()) + 0.5          # cells
        self.rk = int(math.ceil(self.R)) + 2
        self.mR = float(z[self.mask].mean())
        self.zf = np.where(np.isfinite(z), z - self.mR, 0.0)
        self.vf = np.isfinite(z).astype(np.float64)
        self.mf = self.mask
        self.anchor_xy = (grid.x0 + (self.ac + 0.5) * grid.cs,
                          grid.ytop - (self.ar + 0.5) * grid.cs)
        self._rr, self._cc = rr, cc
        self._z = z

    def points(self):
        cs = self.cs
        return RefPoints((self._cc - self.ac) * cs, -(self._rr - self.ar) * cs,
                         self._z[self.mask], cs)

    def template(self, theta_deg):
        """Rotated template on a (2rk+1)^2 grid of offsets (cells) around
        the anchor: returns (T*m, m) with T centred by the reference mean."""
        rk = self.rk
        th = math.radians(theta_deg)
        c, s = math.cos(th), math.sin(th)
        k = np.arange(-rk, rk + 1, dtype=np.float64)
        ex = k[None, :] * np.ones((2 * rk + 1, 1))
        ey = -k[:, None] * np.ones((1, 2 * rk + 1))
        rx = ex * c + ey * s                 # R(-theta) applied to offset
        ry = -ex * s + ey * c
        row = self.ar - ry
        col = self.ac + rx
        coords = np.array([row.ravel(), col.ravel()])
        vf = ndi.map_coordinates(self.vf, coords, order=1, mode="constant", cval=0.0)
        zf = ndi.map_coordinates(self.zf, coords, order=1, mode="constant", cval=0.0)
        ri = np.floor(row.ravel() + 0.5).astype(np.int64)
        ci = np.floor(col.ravel() + 0.5).astype(np.int64)
        inb = (ri >= 0) & (ri < self.mf.shape[0]) & (ci >= 0) & (ci < self.mf.shape[1])
        mk = np.zeros(ri.shape, bool)
        mk[inb] = self.mf[ri[inb], ci[inb]]
        mk &= vf > 1 - 1e-9
        shp = (2 * rk + 1, 2 * rk + 1)
        m = mk.reshape(shp).astype(np.float64)
        T = np.where(mk, zf, 0.0).reshape(shp)
        return T, m


def fft_rmse_scan(Seff, ref: RefTemplate, angles, progress=None,
                  max_tile=MAX_FFT_TILE):
    """RMSE of every grid placement for every angle (FFT, masked).  Returns
    best-over-angle RMSE map and the angle (deg) that achieved it.
    Seff: comparison-grid search surface, NaN = unusable."""
    H, W = Seff.shape
    rk = ref.rk
    valid = np.isfinite(Seff)
    mean0 = float(Seff[valid].mean()) if valid.any() else 0.0
    best = np.full((H, W), np.inf, np.float32)
    bang = np.full((H, W), np.nan, np.float32)
    th_own = max(16, min(H, max_tile - 2 * rk))
    tw_own = max(16, min(W, max_tile - 2 * rk))
    tiles = [(r, c) for r in range(0, H, th_own) for c in range(0, W, tw_own)]
    total = len(tiles) * len(angles)
    done = 0
    # kernels are cheap relative to FFTs; compute once if only one tile
    cache = {}
    for (r0, c0) in tiles:
        r1, c1 = min(r0 + th_own, H), min(c0 + tw_own, W)
        th, tw = r1 - r0, c1 - c0
        # tile with halo, zero padded outside the array
        Ht, Wt = th + 2 * rk, tw + 2 * rk
        S = np.zeros((Ht, Wt))
        V = np.zeros((Ht, Wt))
        a0, a1 = max(r0 - rk, 0), min(r1 + rk, H)
        b0, b1 = max(c0 - rk, 0), min(c1 + rk, W)
        sub_v = valid[a0:a1, b0:b1]
        S[a0 - (r0 - rk):a1 - (r0 - rk), b0 - (c0 - rk):b1 - (c0 - rk)] = \
            np.where(sub_v, Seff[a0:a1, b0:b1] - mean0, 0.0)
        V[a0 - (r0 - rk):a1 - (r0 - rk), b0 - (c0 - rk):b1 - (c0 - rk)] = sub_v
        if not V[rk:rk + th, rk:rk + tw].any():
            done += len(angles)
            continue
        P = (sfft.next_fast_len(Ht, real=True), sfft.next_fast_len(Wt, real=True))
        FS = sfft.rfft2(S, s=P, workers=WORKERS)
        FS2 = sfft.rfft2(S * S, s=P, workers=WORKERS)
        FV = sfft.rfft2(V, s=P, workers=WORKERS)
        K = 2 * rk + 1
        bt = best[r0:r1, c0:c1]
        ba = bang[r0:r1, c0:c1]
        for th_deg in angles:
            if len(tiles) == 1 and th_deg in cache:
                T, m = cache[th_deg]
            else:
                T, m = ref.template(th_deg)
                if len(tiles) == 1:
                    cache[th_deg] = (T, m)
            n = m.sum()
            if n < 4:
                done += 1
                continue
            sT, sT2 = T.sum(), (T * T).sum()
            mp = np.zeros(P)
            mp[:K, :K] = m
            tp = np.zeros(P)
            tp[:K, :K] = T
            Fm = sfft.rfft2(mp, workers=WORKERS).conj()
            Ft = sfft.rfft2(tp, workers=WORKERS).conj()
            A = sfft.irfft2(FS * Fm, s=P, workers=WORKERS)[:th, :tw]
            Q = sfft.irfft2(FS2 * Fm - 2.0 * FS * Ft, s=P, workers=WORKERS)[:th, :tw]
            D = sfft.irfft2(FV * Fm, s=P, workers=WORKERS)[:th, :tw]
            md = (A - sT) / n
            var = (Q + sT2) / n - md * md
            rm = np.sqrt(np.maximum(var, 0.0))
            rm[D < n - 0.5] = np.inf
            better = rm < bt
            bt[better] = rm[better]
            ba[better] = th_deg
            done += 1
            if progress and (done % 4 == 0):
                progress(done / total)
    if progress:
        progress(1.0)
    return best, bang


# --------------------------------------------------------------------------
# Direct (bilinear) scoring and continuous refinement
# --------------------------------------------------------------------------
@dataclass
class RefPoints:
    dx: np.ndarray       # offsets from anchor (map units, east)
    dy: np.ndarray       # offsets from anchor (map units, north)
    z: np.ndarray        # reference elevations
    cs: float            # cell size these points came from

    @property
    def n(self):
        return len(self.z)

    @property
    def R(self):
        return float(np.hypot(self.dx, self.dy).max())

    def subsample(self, stride):
        return RefPoints(self.dx[::stride], self.dy[::stride], self.z[::stride], self.cs)


@dataclass
class Surface:
    arr: np.ndarray      # NaN = unusable (NoData or outside the search area)
    grid: Grid


def sample_bilinear(arr, g: Grid, X, Y):
    col = (X - g.x0) / g.cs - 0.5
    row = (g.ytop - Y) / g.cs - 0.5
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    fc = col - c0
    fr = row - r0
    inb = (r0 >= 0) & (r0 < g.nrows - 1) & (c0 >= 0) & (c0 < g.ncols - 1)
    r0c = np.clip(r0, 0, g.nrows - 2)
    c0c = np.clip(c0, 0, g.ncols - 2)
    v = ((1 - fr) * (1 - fc) * arr[r0c, c0c] + (1 - fr) * fc * arr[r0c, c0c + 1]
         + fr * (1 - fc) * arr[r0c + 1, c0c] + fr * fc * arr[r0c + 1, c0c + 1])
    v[~inb] = np.nan
    return v


def score_poses(pts: RefPoints, surf: Surface, poses):
    """poses: (m,3) [anchor_x, anchor_y, theta_deg] -> (rmse, offset)."""
    poses = np.atleast_2d(poses)
    rm = np.full(len(poses), np.inf)
    off = np.full(len(poses), np.nan)
    for i, (ax, ay, th) in enumerate(poses):
        t = math.radians(th)
        c, s = math.cos(t), math.sin(t)
        X = ax + pts.dx * c - pts.dy * s
        Y = ay + pts.dx * s + pts.dy * c
        v = sample_bilinear(surf.arr, surf.grid, X, Y)
        d = v - pts.z
        if not np.isfinite(d).all():
            continue
        mu = d.mean()
        rm[i] = math.sqrt(float(np.mean((d - mu) ** 2)))
        off[i] = mu
    return rm, off


_STENCIL = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1)
                     for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)], float)


def refine_pose(pts: RefPoints, surf: Surface, pose, s0, smin, rotate=True,
                max_iter=400, big_n=30000):
    """Pattern search over (x, y, theta).  Step s is in units of the
    surface cell size; the angular step is s*cs/R (radians) so the outer
    edge of the footprint moves at most as far as the XY step."""
    cs = surf.grid.cs
    R = max(pts.R, cs)
    stride = max(1, int(math.ceil(pts.n / big_n)))
    pose = np.array(pose, float)
    s = float(s0)
    cur_pts = pts.subsample(stride) if (stride > 1 and s >= 0.25) else pts
    f, off = score_poses(cur_pts, surf, pose)
    f = f[0]
    it = 0
    while s >= smin and it < max_iter:
        it += 1
        use = pts.subsample(stride) if (stride > 1 and s >= 0.25) else pts
        if use is not cur_pts:
            cur_pts = use
            f = score_poses(cur_pts, surf, pose)[0][0]
        step = np.array([cs * s, cs * s, math.degrees(cs * s / R) if rotate else 0.0])
        cand = pose + _STENCIL * step
        if not rotate:
            cand = cand[_STENCIL[:, 2] == 0]
        r, _ = score_poses(cur_pts, surf, cand)
        j = int(np.argmin(r))
        if r[j] < f - 1e-12:
            pose, f = cand[j], r[j]
        else:
            s *= 0.5
    rm, off = score_poses(pts, surf, pose)
    return pose, float(rm[0]), float(off[0])


# --------------------------------------------------------------------------
# Helpers: seeds, overlap, verification
# --------------------------------------------------------------------------
def pick_seeds(best, bang, grid: Grid, n_seeds, sep_cells):
    flat = best.ravel()
    fin = np.flatnonzero(np.isfinite(flat))
    if len(fin) == 0:
        return []
    M = min(len(fin), 400000)
    sel = fin if M == len(fin) else fin[np.argpartition(flat[fin], M - 1)[:M]]
    sel = sel[np.argsort(flat[sel], kind="stable")]
    rr, cc = np.divmod(sel, best.shape[1])
    seeds = []
    sep2 = float(sep_cells) ** 2
    while len(sel) and len(seeds) < n_seeds:
        r, c = rr[0], cc[0]
        seeds.append((grid.x0 + (c + 0.5) * grid.cs, grid.ytop - (r + 0.5) * grid.cs,
                      float(bang[r, c]), float(flat[sel[0]])))
        keep = (rr - r) ** 2 + (cc - c) ** 2 >= sep2
        sel, rr, cc = sel[keep], rr[keep], cc[keep]
    return seeds


def covered_cells(pts: RefPoints, pose, cs):
    ax, ay, th = pose
    t = math.radians(th)
    c, s = math.cos(t), math.sin(t)
    X = ax + pts.dx * c - pts.dy * s
    Y = ay + pts.dx * s + pts.dy * c
    ci = np.floor(X / cs).astype(np.int64)
    ri = np.floor(Y / cs).astype(np.int64)
    return np.unique(ri * 4000003 + ci)


def overlap_fraction(pts, pose_a, pose_b, cs):
    a = covered_cells(pts, pose_a, cs)
    b = covered_cells(pts, pose_b, cs)
    return len(np.intersect1d(a, b, assume_unique=True)) / max(1, min(len(a), len(b)))


def transform_rings(rings, anchor_xy, pose):
    ax, ay, th = pose
    t = math.radians(th)
    c, s = math.cos(t), math.sin(t)
    out = []
    for r in rings:
        r = np.asarray(r, float)[:, :2]
        dx, dy = r[:, 0] - anchor_xy[0], r[:, 1] - anchor_xy[1]
        out.append(np.column_stack([ax + dx * c - dy * s, ay + dx * s + dy * c]))
    return out


def rings_inside(rings, search_rings, fine_cs):
    """Approximate containment test at fine_cs (cell-centre sampling)."""
    x0, y0, x1, y1 = ring_bbox(rings)
    g = snap_grid((x0, y0, x1, y1), fine_cs, 0.0, 0.0, pad_cells=1)
    a = rasterize_rings(rings, g)
    b = rasterize_rings(search_rings, g)
    return not (a & ~b).any()


def narrow_fraction(mask, min_width_cells=3):
    """Fraction of mask cells in parts thinner than min_width_cells."""
    rad = min_width_cells / 2.0
    k = int(math.ceil(rad))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    st = (xx ** 2 + yy ** 2) <= rad ** 2 + 1e-9
    m = np.pad(mask, k + 1)
    opened = ndi.binary_opening(m, structure=st)
    return 1.0 - opened.sum() / max(1, m.sum())


def nice_cell_size(x):
    """Round up to the 1-2-5 series."""
    if x <= 0:
        return x
    e = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if x <= m * e * (1 + 1e-9):
            return m * e
    return 10 * e


def suggest_cell_size(native_cs, ref_area, search_area,
                      target_ref_radius_cells=10.0, max_search_cells=4e6):
    """Default comparison cell size (m), from tests/benchmark.py: a reference
    footprint ~10 cells in (equivalent) radius found the true site with the
    same final accuracy as a native search at ~1/50 of the time, because the
    native-resolution refinement re-scores the shortlist.  The search grid is
    kept under ~4 M cells, but never coarser than 6 cells of reference radius
    (the shape would no longer be captured)."""
    r_eq = math.sqrt(max(ref_area, 1e-9) / math.pi)
    cs = r_eq / target_ref_radius_cells
    cs = max(cs, math.sqrt(max(search_area, 1e-9) / max_search_cells))
    cs = min(cs, r_eq / 6.0)
    cs = max(cs, native_cs)
    return float(nice_cell_size(cs / native_cs) * native_cs)


# --------------------------------------------------------------------------
# Results and main driver
# --------------------------------------------------------------------------
@dataclass
class Match:
    rank: int
    rmse: float                 # final-resolution RMSE (m)
    offset: float               # mean(candidate - reference) (m)
    angle: float                # deg CCW, [0, 360)
    x: float                    # anchor position (map units)
    y: float
    final_cs: float
    search_cs: float
    coarse_rmse: float
    rings: list = field(default_factory=list)
    checks: str = ""


@dataclass
class SearchResult:
    matches: list
    anchor_xy: tuple
    search_cs: float
    final_cs: float
    angle_step_coarse: float
    n_angles: int
    mode: str
    notes: list
    warnings: list
    timings: dict
    ref_rings: list = field(default_factory=list)



# --------------------------------------------------------------------------
# Multi-resolution pyramid
# --------------------------------------------------------------------------
PYRAMID_FACTOR = 2.5
PYRAMID_MIN_RADIUS_CELLS = 5.0


def pyramid_levels(cs, n_ref_cells, max_levels=4):
    """Coarser comparison cell sizes (coarsest first) above `cs`, stopping when
    the reference footprint would be < ~5 cells in equivalent radius."""
    r_eq_m = math.sqrt(n_ref_cells / math.pi) * cs
    out = []
    c = cs
    while len(out) < max_levels - 1:
        c = c * PYRAMID_FACTOR
        if r_eq_m / c < PYRAMID_MIN_RADIUS_CELLS:
            break
        out.append(c)
    return out[::-1]


def convert_pose(pose, a_from, a_to):
    """Same physical placement expressed for a different anchor point."""
    ax, ay, th = pose
    t = math.radians(th)
    c, s = math.cos(t), math.sin(t)
    dx, dy = a_to[0] - a_from[0], a_to[1] - a_from[1]
    return (ax + dx * c - dy * s, ay + dx * s + dy * c, th)


def _coarse_step_for(R_cells, coarse_step=None):
    step = coarse_step or float(np.clip(math.degrees(1.5 / R_cells), 1.0, 20.0))
    return 360.0 / int(math.ceil(360.0 / step - 1e-9))


class _Level:
    pass


def _build_level(dem, cs, ref_rings, search_rings, ref_bb, srch_bb, min_cov):
    geo = dem.geo
    lv = _Level()
    lv.cs = cs
    rgrid = snap_grid(ref_bb, cs, geo.x0, geo.ytop, pad_cells=1)
    zref = read_aggregated(dem, rgrid, min_cov)
    rmask = rasterize_rings(ref_rings, rgrid) & np.isfinite(zref)
    if rmask.sum() < 12:
        return None
    lv.ref = RefTemplate(zref, rmask, rgrid)
    lv.grid = snap_grid(srch_bb, cs, geo.x0, geo.ytop, pad_cells=1)
    S = read_aggregated(dem, lv.grid, min_cov)
    inside = rasterize_rings(search_rings, lv.grid)
    Seff = np.where(inside & np.isfinite(S), S, np.nan)
    usable = ndi.binary_erosion(np.isfinite(Seff), structure=np.ones((3, 3), bool), border_value=0)
    lv.S = np.where(usable, Seff, np.nan)
    return lv


def _nms(cands, min_dist, keep):
    cands = sorted(cands, key=lambda c: c[3])
    out = []
    for c in cands:
        if all(math.hypot(c[0] - o[0], c[1] - o[1]) >= min_dist for o in out):
            out.append(c)
            if len(out) >= keep:
                break
    return out


def pyramid_seeds(dem, ref_rings, search_rings, ref_bb, srch_bb, levels_cs, ref_fin, grid_fin,
                  S_fin, rotate, coarse_step, n_short, n_results, min_cov, prog, say, scan_frac):
    """Coarse-to-fine candidate generation.  Level 0 (coarsest) scans every
    placement; every finer level only re-scores the surviving candidates in
    a small window (+/- 1.5 coarse cells) and a few angles around the
    inherited angle.  Returns seeds [(x, y, theta, score)] in the *final*
    level's anchor frame."""
    levels = []
    for c in levels_cs:
        lv = _build_level(dem, c, ref_rings, search_rings, ref_bb, srch_bb, min_cov)
        if lv is not None and np.isfinite(lv.S).any():
            levels.append(lv)
    fin = _Level()
    fin.cs, fin.ref, fin.grid, fin.S = ref_fin.cs, ref_fin, grid_fin, S_fin
    levels.append(fin)
    L = len(levels)
    K0 = max(300, 30 * n_results)
    Ks = [int(round(K0 * (n_short / K0) ** (i / max(1, L - 1)))) for i in range(L)]
    Ks[-1] = n_short
    steps = [_coarse_step_for(lv.ref.R, coarse_step) if rotate else 0.0 for lv in levels]
    lv0 = levels[0]
    angles = np.arange(int(round(360.0 / steps[0]))) * steps[0] if rotate else np.array([0.0])
    say("Pyramid: %s; level 0 scans %d x %d cells x %d angles (%.3g deg), keeps %d candidates."
        % (", ".join("%g m (R=%.1f cells)" % (lv.cs, lv.ref.R) for lv in levels),
           lv0.grid.nrows, lv0.grid.ncols, len(angles), steps[0], Ks[0]))

    def p0(f):
        prog(0.05 + scan_frac * 0.5 * f, "Coarse scan at %g m: %d angles over %d x %d cells (%d%%)"
             % (lv0.cs, len(angles), lv0.grid.nrows, lv0.grid.ncols, int(100 * f)))

    if L == 1:                       # all coarse levels dropped: plain scan
        best, bang = fft_rmse_scan(lv0.S, lv0.ref, angles, progress=p0)
    else:
        best, bang = fft_rmse_scan(lv0.S, lv0.ref, angles, progress=p0)
    if not np.isfinite(best).any():
        raise ValueError("No placement of the reference footprint fits entirely inside the search "
                         "polygon on valid DTM data at the coarsest pyramid level (%g m)." % lv0.cs)
    sep0 = max(2.0, 0.5 * math.sqrt(lv0.ref.n / math.pi))
    cands = pick_seeds(best, bang, lv0.grid, Ks[0], sep0)
    say("  level %g m: %d candidates kept, best score %.3f m"
        % (lv0.cs, len(cands), cands[0][3] if cands else float("nan")))
    for k in range(1, L):
        prev, lv = levels[k - 1], levels[k]
        hw = int(math.ceil(1.5 * prev.cs / lv.cs)) + 1
        m = int(math.ceil(steps[k - 1] / steps[k])) if (rotate and steps[k] > 0) else 0
        g, ref = lv.grid, lv.ref
        rk = ref.rk
        out = []
        for ci, (x, y, th, sc) in enumerate(cands):
            ax, ay, _ = convert_pose((x, y, th), prev.ref.anchor_xy, ref.anchor_xy)
            i = int(round((g.ytop - ay) / g.cs - 0.5))
            j = int(round((ax - g.x0) / g.cs - 0.5))
            r0, r1 = max(i - hw - rk, 0), min(i + hw + rk + 1, g.nrows)
            c0, c1 = max(j - hw - rk, 0), min(j + hw + rk + 1, g.ncols)
            if r1 - r0 < 2 * rk + 1 or c1 - c0 < 2 * rk + 1:
                continue
            crop = lv.S[r0:r1, c0:c1]
            if not np.isfinite(crop).any():
                continue
            if rotate:
                angs = (th + np.arange(-m, m + 1) * steps[k]) % 360.0
            else:
                angs = np.array([0.0])
            bm, ba = fft_rmse_scan(crop, ref, angs)
            q = int(np.argmin(bm))
            if not np.isfinite(bm.flat[q]):
                continue
            ri, cj = divmod(q, bm.shape[1])
            out.append((g.x0 + (c0 + cj + 0.5) * g.cs, g.ytop - (r0 + ri + 0.5) * g.cs,
                        float(ba[ri, cj]), float(bm.flat[q])))
            if ci % 10 == 0:
                prog(0.05 + scan_frac * (0.5 + 0.5 * ((k - 1) + ci / max(1, len(cands))) / (L - 1)),
                     "Re-scoring %d candidates at %g m (level %d/%d)" % (len(cands), lv.cs, k, L - 1))
        cands = _nms(out, max(2.0, 0.5 * math.sqrt(ref.n / math.pi)) * lv.cs, Ks[k])
        say("  level %g m: %d candidates kept, best score %.3f m"
            % (lv.cs, len(cands), cands[0][3] if cands else float("nan")))
        if not cands:
            raise ValueError("The pyramid search lost all candidates at %g m." % lv.cs)
    return cands


def find_similar_terrain(dem, ref_rings, search_rings, comparison_cs=None,
                         refine_native=True, n_results=5, rotate=True,
                         exhaustive=False, coarse_step=None, exhaustive_step=None,
                         shortlist=None, max_overlap=0.2, min_cov=0.5, pyramid=True,
                         progress=None, log=None):
    """Main entry point.  `dem` provides .geo (native Grid) and
    .window(xmin, ymin, xmax, ymax) -> (array with NaN, Grid)."""
    T0 = time.time()
    timings = {}
    warnings, notes = [], []

    def say(msg):
        if log:
            log(msg)

    def prog(frac, msg=None):
        if progress:
            progress(min(max(frac, 0.0), 1.0), msg)

    geo = dem.geo
    ncs = geo.cs
    ref_rings = [np.asarray(r, float)[:, :2] for r in ref_rings]
    search_rings = [np.asarray(r, float)[:, :2] for r in search_rings]

    # ---- comparison resolution -------------------------------------------
    ref_bb, srch_bb = ring_bbox(ref_rings), ring_bbox(search_rings)
    if comparison_cs is None:
        ref_area = rasterize_rings(ref_rings, snap_grid(ref_bb, ncs, geo.x0, geo.ytop)).sum() * ncs ** 2
        s_area = (srch_bb[2] - srch_bb[0]) * (srch_bb[3] - srch_bb[1])
        comparison_cs = suggest_cell_size(ncs, ref_area, s_area)
        notes.append("Comparison cell size not given: automatic choice %g m." % comparison_cs)
    cs = float(comparison_cs)
    if cs < ncs * (1 - 1e-6):
        warnings.append("Comparison cell size %g m is finer than the native %g m; using native."
                        % (cs, ncs))
        cs = ncs
    two_stage = bool(refine_native) and cs > ncs * (1 + 1e-6)

    # ---- reference on comparison grid ------------------------------------
    prog(0.0, "Reading and aggregating DTM to %g m" % cs)
    t = time.time()
    rgrid = snap_grid(ref_bb, cs, geo.x0, geo.ytop, pad_cells=1)
    zref = read_aggregated(dem, rgrid, min_cov)
    rmask_poly = rasterize_rings(ref_rings, rgrid)
    rmask = rmask_poly & np.isfinite(zref)
    if rmask_poly.sum() < 12:
        raise ValueError("Reference polygon covers only %d cells at %g m - choose a smaller "
                         "comparison cell size." % (rmask_poly.sum(), cs))
    if rmask.sum() < 0.5 * rmask_poly.sum():
        raise ValueError("More than half of the reference polygon is NoData in the DTM.")
    if rmask.sum() < rmask_poly.sum():
        warnings.append("%d reference cells are NoData and were excluded from the comparison."
                        % (rmask_poly.sum() - rmask.sum()))
    ref = RefTemplate(zref, rmask, rgrid)
    nf = narrow_fraction(rmask, 3)
    if nf > 0.10:
        warnings.append("%.0f%% of the reference polygon is narrower than 3 cells at %g m; its "
                        "shape is poorly captured. Use a smaller comparison cell size."
                        % (100 * nf, cs))
    if ref.n < 80:
        warnings.append("Reference polygon has only %d cells at %g m; matching may be unreliable."
                        % (ref.n, cs))

    sgrid = snap_grid(srch_bb, cs, geo.x0, geo.ytop, pad_cells=1)
    ncell = float(sgrid.nrows) * float(sgrid.ncols)
    if ncell > MAX_SEARCH_CELLS:
        need = cs * math.sqrt(ncell / MAX_SEARCH_CELLS)
        raise ValueError(
            "The search polygon spans %d x %d cells at the %g m comparison cell size (%.3g cells; "
            "limit %.3g). Use a comparison cell size of about %g m or larger, or a smaller search "
            "polygon. Check also that the search polygon is in the same units/area as the DTM "
            "(its extent is %.4g x %.4g map units)."
            % (sgrid.ncols, sgrid.nrows, cs, ncell, MAX_SEARCH_CELLS, math.ceil(need),
               srch_bb[2] - srch_bb[0], srch_bb[3] - srch_bb[1]))
    S = read_aggregated(dem, sgrid, min_cov)
    inside = rasterize_rings(search_rings, sgrid)
    Seff = np.where(inside & np.isfinite(S), S, np.nan)
    usable = ndi.binary_erosion(np.isfinite(Seff), structure=np.ones((3, 3), bool),
                                border_value=0)
    Seff_fft = np.where(usable, Seff, np.nan)
    if not usable.any():
        raise ValueError("The search polygon contains no valid DTM cells at %g m." % cs)
    timings["read_aggregate"] = time.time() - t

    # ---- angular grid -----------------------------------------------------
    R_cells = ref.R
    if not rotate:
        angles = np.array([0.0])
        step = 0.0
        mode = "no rotation (0 deg only)"
    elif exhaustive:
        step = exhaustive_step or float(np.clip(math.degrees(0.5 / R_cells), 0.25, 5.0))
        nang = int(math.ceil(360.0 / step - 1e-9))
        step = 360.0 / nang
        angles = np.arange(nang) * step
        mode = ("EXHAUSTIVE: every placement on the %g m grid x %d angles (%.4g deg increments) "
                "- scores exact for that grid/increment; positions/angles are then refined "
                "continuously" % (cs, nang, step))
    else:
        step = coarse_step or float(np.clip(math.degrees(1.5 / R_cells), 1.0, 20.0))
        nang = int(math.ceil(360.0 / step - 1e-9))
        step = 360.0 / nang
        angles = np.arange(nang) * step
        mode = "coarse-to-fine: %d coarse angles (%.3g deg), then continuous refinement" % (nang, step)
    say("Comparison grid %g m: search %d x %d cells, reference %d cells (radius %.1f cells)."
        % (cs, sgrid.nrows, sgrid.ncols, ref.n, R_cells))
    n_short = int(shortlist or max(20, 4 * n_results))
    sep = max(3.0, math.sqrt(ref.n / math.pi))
    levels_cs = [] if (exhaustive or not pyramid) else pyramid_levels(cs, ref.n)
    scan_frac = 0.6
    t = time.time()
    if levels_cs:
        mode = ("multi-resolution pyramid %s m (coarse screening, then local re-scoring at each "
                "finer level), then continuous refinement"
                % " > ".join("%g" % c for c in levels_cs + [cs]))
        say("Rotation search: " + mode)
        seeds = pyramid_seeds(dem, ref_rings, search_rings, ref_bb, srch_bb, levels_cs,
                              ref, sgrid, Seff_fft, rotate, coarse_step, n_short, n_results,
                              min_cov, prog, say, scan_frac)
        step = 0.0
    else:
        say("Rotation search: " + mode)
        est = len(angles) * ncell * 2.5e-7      # calibrated on the 1500x1500 benchmark
        say("Estimated scan time: ~%s (%d angles x %.3g cells); refinement adds more."
            % (("%.0f s" % est) if est < 120 else ("%.0f min" % (est / 60)), len(angles), ncell))
        if est > 600:
            warnings.append("Estimated scan time is %.0f min. Enable the multi-resolution pyramid, "
                            "or use a larger comparison cell size (native refinement on keeps the "
                            "final accuracy)." % (est / 60))

        def scan_prog(f):
            prog(0.05 + scan_frac * f,
                 "Scanning %d rotation(s) over %d x %d cells (%d%%)"
                 % (len(angles), sgrid.nrows, sgrid.ncols, int(100 * f)))

        best, bang = fft_rmse_scan(Seff_fft, ref, angles, progress=scan_prog)
        if not np.isfinite(best).any():
            raise ValueError("No placement of the reference footprint fits entirely inside the "
                             "search polygon on valid DTM data at %g m." % cs)
        seeds = pick_seeds(best, bang, sgrid, n_short, sep)
    timings["scan"] = time.time() - t
    say("Shortlist: %d distinct seeds (requested results: %d)." % (len(seeds), n_results))

    # ---- continuous refinement at comparison resolution --------------------
    t = time.time()
    pts_c = ref.points()
    surf_c = Surface(Seff, sgrid)
    anchor = ref.anchor_xy
    cands = []
    for i, (x, y, th, sc) in enumerate(seeds):
        pose, rm, off = refine_pose(pts_c, surf_c, (x, y, th), 1.0, 0.05, rotate)
        if np.isfinite(rm):
            cands.append(dict(pose=pose, rmse=rm, offset=off, coarse=rm, cs=cs))
        prog(0.05 + scan_frac + 0.15 * (i + 1) / max(1, len(seeds)),
             "Refining shortlist at %g m (%d/%d)" % (cs, i + 1, len(seeds)))
    timings["refine_search_res"] = time.time() - t
    cands = _dedupe(cands, pts_c, cs, max_overlap, sep * cs * 0.5, keep=None)

    # ---- optional refinement at native resolution --------------------------
    if two_stage:
        t = time.time()
        wr = dem.window(ref_bb[0] - ncs, ref_bb[1] - ncs, ref_bb[2] + ncs, ref_bb[3] + ncs)
        zr_n, gr_n = wr
        mk_n = rasterize_rings(ref_rings, gr_n) & np.isfinite(zr_n)
        poly_n = rasterize_rings(ref_rings, gr_n)
        if mk_n.sum() < poly_n.sum():
            warnings.append("%d native reference cells are NoData and were excluded."
                            % (poly_n.sum() - mk_n.sum()))
        rr, cc = np.nonzero(mk_n)
        xs, ys = gr_n.centers()
        pts_n = RefPoints(xs[cc] - anchor[0], ys[rr] - anchor[1], zr_n[mk_n], ncs)
        Rn = pts_n.R
        new = []
        for i, cd in enumerate(cands):
            ax, ay, _ = cd["pose"]
            rad = Rn + 4 * cs
            arr, g = dem.window(ax - rad, ay - rad, ax + rad, ay + rad)
            if arr is None:
                continue
            ins = rasterize_rings(search_rings, g)
            ins = ndi.binary_erosion(ins, structure=np.ones((3, 3), bool), border_value=0)
            surf_n = Surface(np.where(ins & np.isfinite(arr), arr, np.nan), g)
            pose, rm, off = refine_pose(pts_n, surf_n, cd["pose"], 0.5 * cs / ncs, 0.05, rotate)
            if np.isfinite(rm):
                new.append(dict(pose=pose, rmse=rm, offset=off, coarse=cd["rmse"], cs=ncs))
            prog(0.05 + scan_frac + 0.15 + 0.15 * (i + 1) / max(1, len(cands)),
                 "Refining shortlist at native %g m (%d/%d)" % (ncs, i + 1, len(cands)))
        timings["refine_native"] = time.time() - t
        cands = new
        final_cs = ncs
    else:
        final_cs = cs
        pts_n, Rn = pts_c, ref.R * cs
    cands = _dedupe(cands, pts_c, cs, max_overlap, sep * cs * 0.5, keep=None)

    # ---- verification and final selection ---------------------------------
    matches = []
    dropped = 0
    for cd in cands:
        if len(matches) >= n_results:
            break
        ax, ay, th = cd["pose"]
        th = th % 360.0
        rings = transform_rings(ref_rings, anchor, (ax, ay, th))
        ok_inside = rings_inside(rings, search_rings, final_cs / 2.0)
        rad = Rn + 2 * final_cs
        arr, g = dem.window(ax - rad, ay - rad, ax + rad, ay + rad)
        ok_valid = False
        if arr is not None:
            cells = rasterize_rings(rings, g)
            ok_valid = bool(cells.any() and np.isfinite(arr[cells]).all())
        if not (ok_inside and ok_valid):
            dropped += 1
            continue
        matches.append(Match(len(matches) + 1, cd["rmse"], cd["offset"], th, ax, ay,
                             cd["cs"], cs, cd["coarse"], rings,
                             "inside search: yes; all cells valid: yes"))
    if dropped:
        warnings.append("%d candidate(s) failed the inside-search / valid-data verification "
                        "and were discarded." % dropped)
    if len(matches) < n_results:
        warnings.append("Only %d distinct matches were found (%d requested)."
                        % (len(matches), n_results))
    timings["total"] = time.time() - T0
    prog(1.0, "Done")
    return SearchResult(matches, anchor, cs, final_cs, step, len(angles), mode, notes,
                        warnings, timings, ref_rings)


def _dedupe(cands, pts, cs, max_overlap, min_dist, keep=None):
    cands = sorted(cands, key=lambda d: d["rmse"])
    out = []
    for cd in cands:
        dup = False
        for o in out:
            dist = math.hypot(cd["pose"][0] - o["pose"][0], cd["pose"][1] - o["pose"][1])
            if dist < min_dist or overlap_fraction(pts, cd["pose"], o["pose"], cs) > max_overlap:
                dup = True
                break
        if not dup:
            out.append(cd)
    return out if keep is None else out[:keep]


# --------------------------------------------------------------------------
# Residual raster for a match (final resolution)
# --------------------------------------------------------------------------
def residual_grid(dem, ref_rings, anchor_xy, match: Match):
    """Signed residual (candidate - reference - offset) on the native grid,
    inside the matched polygon.  Returns (array with NaN outside, Grid)."""
    ncs = dem.geo.cs
    fcs = match.final_cs
    rb = ring_bbox(ref_rings)
    zr, gr = dem.window(rb[0] - ncs, rb[1] - ncs, rb[2] + ncs, rb[3] + ncs)
    mb = ring_bbox(match.rings)
    arr, g = dem.window(mb[0] - 2 * ncs, mb[1] - 2 * ncs, mb[2] + 2 * ncs, mb[3] + 2 * ncs)
    cells = rasterize_rings(match.rings, g)
    xs, ys = g.centers()
    rr, cc = np.nonzero(cells)
    X, Y = xs[cc] - match.x, ys[rr] - match.y
    t = math.radians(match.angle)
    c, s = math.cos(t), math.sin(t)
    # preimage in reference space: R(-theta)
    px = anchor_xy[0] + X * c + Y * s
    py = anchor_xy[1] - X * s + Y * c
    zref = sample_bilinear(zr, gr, px, py)
    res = np.full(g.shape, np.nan)
    res[rr, cc] = arr[rr, cc] - zref - match.offset
    return res, g
