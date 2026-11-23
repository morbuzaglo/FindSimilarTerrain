"""Core tests (no ArcGIS needed).  Run:  python tests/test_core.py"""
import math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import terrain_core as tc
import synth

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + ("  | " + detail if detail else ""), flush=True)


def test_score_semantics():
    """+100 m copy scores ~0; twice the relief is penalised; slope kept."""
    rng = np.random.default_rng(0)
    z = np.cumsum(np.cumsum(rng.standard_normal((60, 60)), 0), 1) * 0.2
    g = tc.Grid(0, 600, 10.0, 60, 60)
    ref = z[10:25, 10:25]
    rmask = np.ones_like(ref, bool)
    sg = tc.Grid(0, 600, 10, 60, 60)
    # a candidate region = same terrain +100
    S = z.copy()
    S[35:50, 35:50] = ref + 100
    # twice relief version
    S2 = z.copy()
    S2[35:50, 35:50] = ref.mean() + 2 * (ref - ref.mean())
    rg = tc.Grid(100, 600 - 100, 10, 15, 15)
    from terrain_core import RefPoints
    cc, rr = np.meshgrid(np.arange(15), np.arange(15))
    pts = RefPoints((cc.ravel() - 7) * 10.0, -(rr.ravel() - 7) * 10.0, ref.ravel(), 10.0)
    pose = (35.5 * 10 + 70 - 70 + 0, 0, 0)
    ax = (35 + 7 + 0.5) * 10
    ay = 600 - (35 + 7 + 0.5) * 10
    r1, o1 = tc.score_poses(pts, tc.Surface(S, sg), [(ax, ay, 0.0)])
    r2, o2 = tc.score_poses(pts, tc.Surface(S2, sg), [(ax, ay, 0.0)])
    report("score: +100 m copy -> RMSE~0, offset~100", r1[0] < 1e-9 and abs(o1[0] - 100) < 1e-9,
           "rmse=%.2e offset=%.4f" % (r1[0], o1[0]))
    report("score: twice-relief copy penalised", abs(r2[0] - ref.std()) < 1e-6 and r2[0] > 0.5,
           "rmse=%.3f (expected std(ref)=%.3f)" % (r2[0], ref.std()))


def test_rasterizer():
    g = tc.Grid(0, 100, 1.0, 100, 100)
    outer = np.array([[10, 10], [10, 90], [90, 90], [90, 10], [10, 10]], float)
    hole = np.array([[40, 40], [60, 40], [60, 60], [40, 60], [40, 40]], float)
    m = tc.rasterize_rings([outer, hole], g)
    expect = 80 * 80 - 20 * 20
    report("rasterizer: polygon with hole", m.sum() == expect, "%d cells (expected %d)" % (m.sum(), expect))


def test_aggregation():
    rng = np.random.default_rng(1)
    a = rng.random((100, 100)) * 10
    a[20:30, 20:30] = np.nan
    geo = tc.Grid(0, 200, 2.0, 100, 100)
    g5 = tc.Grid(0, 200, 5.0, 40, 40)
    out = tc.aggregate(a, geo, g5, 0.5)
    # cell (0,0): native 2 m cells overlap: 5 m = 2.5 native cells: weights 1,1,0.5
    w = np.array([1, 1, 0.5])
    exp = (a[:3, :3] * np.outer(w, w)).sum() / np.outer(w, w).sum()
    g10 = tc.Grid(0, 200, 10.0, 20, 20)
    o10 = tc.aggregate(a, geo, g10, 0.5)
    exp10 = a[:5, :5].mean()
    nodata_ok = np.isnan(o10[5, 5]) or True
    hole_cell_valid = np.isfinite(o10[4, 4])   # cell covering 40..50 native cells 20..25 -> fully NaN
    report("aggregation: area-weighted 5 m and 10 m block mean",
           abs(out[0, 0] - exp) < 1e-9 and abs(o10[0, 0] - exp10) < 1e-9 and not np.isfinite(o10[4, 4]),
           "5m err=%.1e 10m err=%.1e" % (abs(out[0, 0] - exp), abs(o10[0, 0] - exp10)))


def test_fft_vs_bruteforce():
    """FFT masked scores vs per-placement brute force (same template)."""
    rng = np.random.default_rng(5)
    n = 48
    z = np.cumsum(np.cumsum(rng.standard_normal((n, n)), 0), 1) * 0.3
    z[10:14, 30:34] = np.nan                       # NoData hole
    grid = tc.Grid(0, n * 10.0, 10.0, n, n)
    # irregular masked reference with a hole, cut out of z elsewhere
    ref_z = z[24:41, 4:21].copy()
    rg = tc.Grid(0, 0, 10, ref_z.shape[0], ref_z.shape[1])
    yy, xx = np.mgrid[0:17, 0:17]
    mask = ((yy - 8) ** 2 / 64 + (xx - 8) ** 2 / 49 < 1) & ~(((yy - 6) ** 2 + (xx - 9) ** 2) < 6)
    ref = tc.RefTemplate(ref_z, mask, rg)
    angles = [0.0, 37.0, 90.0, 211.0]
    best, bang = tc.fft_rmse_scan(z, ref, angles)
    maxerr = 0.0
    nchecked = 0
    brute_best = np.full((n, n), np.inf)
    for th in angles:
        T, m = ref.template(th)
        rk = ref.rk
        nn = m.sum()
        mk = m > 0
        zc = z - np.nanmean(z)
        for i in range(rk, n - rk):
            for j in range(rk, n - rk):
                win = zc[i - rk:i + rk + 1, j - rk:j + rk + 1]
                if not np.isfinite(win[mk]).all():
                    continue
                d = win[mk] - T[mk]
                v = math.sqrt(np.mean((d - d.mean()) ** 2))
                brute_best[i, j] = min(brute_best[i, j], v)
    both = np.isfinite(brute_best) & np.isfinite(best)
    maxerr = np.abs(brute_best[both] - best[both]).max()
    same_valid = np.array_equal(np.isfinite(brute_best[rk:n - rk, rk:n - rk]),
                                np.isfinite(best[rk:n - rk, rk:n - rk]))
    report("FFT scan == brute force over %d placements x %d angles" % (both.sum(), len(angles)),
           maxerr < 1e-6 and same_valid, "max |diff|=%.2e m, validity identical=%s" % (maxerr, same_valid))


def run_known_site(cs_cmp, refine_native, label, n=1000, exhaustive=False, targets=None, verbose=True):
    d = synth.make_dem(n=n)
    geo = d["geo"]
    dem = tc.ArrayDEM(d["z"], geo)
    refc = d["truth"]["ref_center"]
    ref_world = synth.world_rings(d["ref_local"], refc[0], refc[1], 0.0)
    ext = (0, 0, n * geo.cs, n * geo.cs)
    srch = synth.search_rings_for(ext, hole_center=d["truth"]["hidden"][0][:2], exclude_ref=(refc[0], refc[1], 230))
    t0 = time.time()
    res = tc.find_similar_terrain(dem, ref_world, srch, comparison_cs=cs_cmp,
                                  refine_native=refine_native, n_results=5, exhaustive=exhaustive)
    el = time.time() - t0
    return d, res, el


def test_known_site():
    d, res, el = run_known_site(10.0, True, "10m->2m")
    ref_anchor_local = None
    # anchor in local coordinates: anchor world - ref center
    refc = d["truth"]["ref_center"]
    a_local = (res.anchor_xy[0] - refc[0], res.anchor_xy[1] - refc[1])
    t = d["truth"]["targets"][0]
    true_xy = synth.true_anchor(a_local, t[0], t[1], t[2])
    m = res.matches[0]
    dpos = math.hypot(m.x - true_xy[0], m.y - true_xy[1])
    dang = abs((m.angle - t[2] + 180) % 360 - 180)
    report("known site: position recovered (10 m search -> 2 m score)", dpos < 3.0,
           "found (%.1f, %.1f) true (%.1f, %.1f) err=%.2f m" % (m.x, m.y, true_xy[0], true_xy[1], dpos))
    report("known site: angle recovered", dang < 0.5, "found %.3f deg, true %.1f deg, err=%.3f" % (m.angle, t[2], dang))
    report("known site: offset ~ +100 m", abs(m.offset - 100.0) < 0.5, "offset=%.3f m" % m.offset)
    report("known site: RMSE small (interpolation error only)", m.rmse < 0.5,
           "RMSE=%.4f m at %.0f m scoring cell, search %.0f m, %.1fs" % (m.rmse, m.final_cs, m.search_cs, el))
    # decoy: same shape x2 relief must rank below the true site and have large RMSE
    dc = d["truth"]["decoys"][0]
    dec = [x for x in res.matches[1:] if math.hypot(x.x - synth.true_anchor(a_local, dc[0], dc[1], dc[2])[0],
                                                     x.y - synth.true_anchor(a_local, dc[0], dc[1], dc[2])[1]) < 15]
    if dec:
        report("decoy (2x relief) is penalised and ranked below the true site",
               dec[0].rmse > 20 * m.rmse and dec[0].rank > 1,
               "decoy rank %d RMSE %.2f m vs true %.3f m" % (dec[0].rank, dec[0].rmse, m.rmse))
    else:
        report("decoy (2x relief) not in top 5 (penalised)", True, "not returned among top 5")
    # hole: nothing may overlap the excluded hidden copy
    h = d["truth"]["hidden"][0]
    hole_ok = all(math.hypot(x.x - synth.true_anchor(a_local, h[0], h[1], h[2])[0],
                             x.y - synth.true_anchor(a_local, h[0], h[1], h[2])[1]) > 100 for x in res.matches)
    report("copy hidden inside search-polygon hole is not returned", hole_ok)
    # containment independent check
    srch = synth.search_rings_for((0, 0, 2000, 2000), hole_center=h[:2], exclude_ref=(300, 300, 230))
    ok = all(tc.rings_inside(x.rings, srch, 0.5) for x in res.matches)
    report("all %d returned polygons inside search area (fine 0.5 m raster check, holes honoured)" % len(res.matches), ok)
    # distinctness
    pts = None
    dd = [math.hypot(a.x - b.x, a.y - b.y) for i, a in enumerate(res.matches) for b in res.matches[i + 1:]]
    report("results are distinct sites", (min(dd) > 60) if dd else True, "min centre separation %.0f m" % (min(dd) if dd else 0))
    print("   ranking:", [(x.rank, round(x.rmse, 3), round(x.offset, 1), round(x.angle, 1)) for x in res.matches])
    return res


def test_exhaustive_vs_default():
    """Small grid: default coarse-to-fine result == exhaustive result."""
    d = synth.make_dem(n=500, targets=((650.0, 650.0, 73.0, 100.0, 1.0),), decoys=((300.0, 700.0, 200.0, -40.0, 2.0),),
                       hidden=((750.0, 250.0, 20.0, 50.0, 1.0),), ref_center=(250.0, 250.0))
    geo = d["geo"]
    dem = tc.ArrayDEM(d["z"], geo)
    rw = synth.world_rings(d["ref_local"], 250, 250, 0.0)
    srch = synth.search_rings_for((0, 0, 1000, 1000), hole_center=(750, 250), exclude_ref=(250, 250, 230))
    t0 = time.time()
    a = tc.find_similar_terrain(dem, rw, srch, comparison_cs=10.0, refine_native=True, n_results=3)
    ta = time.time() - t0
    t0 = time.time()
    b = tc.find_similar_terrain(dem, rw, srch, comparison_cs=10.0, refine_native=True, n_results=3, exhaustive=True)
    tb = time.time() - t0
    ma, mb = a.matches[0], b.matches[0]
    same = math.hypot(ma.x - mb.x, ma.y - mb.y) < 1.0 and abs((ma.angle - mb.angle + 180) % 360 - 180) < 0.3
    report("optimized (coarse-to-fine) == exhaustive on small grid", same,
           "default %.1fs: (%.1f,%.1f) %.2f deg rmse %.3f | exhaustive %.1fs: (%.1f,%.1f) %.2f deg rmse %.3f"
           % (ta, ma.x, ma.y, ma.angle, ma.rmse, tb, mb.x, mb.y, mb.angle, mb.rmse))
    print("   exhaustive mode text:", b.mode)


def test_nodata_and_warnings():
    d = synth.make_dem(n=400, targets=((500.0, 500.0, 73.0, 100.0, 1.0),), decoys=(), hidden=(), ref_center=(200., 200.))
    z = d["z"].copy()
    z[200:260, 200:260] = np.nan                      # NoData patch
    dem = tc.ArrayDEM(z, d["geo"])
    rw = synth.world_rings(d["ref_local"], 200, 200, 0.0)
    srch = synth.search_rings_for((0, 0, 800, 800), (900, 900), exclude_ref=(200, 200, 230))
    res = tc.find_similar_terrain(dem, rw, srch, comparison_cs=20.0, refine_native=True, n_results=3)
    arr = z
    ok = True
    for m in res.matches:
        g = d["geo"]
        cells = tc.rasterize_rings(m.rings, g)
        ok &= bool(np.isfinite(arr[cells]).all())
    report("returned polygons never touch NoData", ok)
    res2 = tc.find_similar_terrain(dem, rw, srch, comparison_cs=30.0, refine_native=False, n_results=1)
    wtxt = " | ".join(res2.warnings)
    report("narrow-part warning at a too-coarse cell size", "narrower than 3 cells" in wtxt or "only" in wtxt, wtxt[:160])


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    test_score_semantics()
    test_rasterizer()
    test_aggregation()
    test_fft_vs_bruteforce()
    test_exhaustive_vs_default()
    test_known_site()
    test_nodata_and_warnings()
    bad = [r for r in RESULTS if not r[1]]
    print("\n%d tests, %d failed" % (len(RESULTS), len(bad)))
    sys.exit(1 if bad else 0)
