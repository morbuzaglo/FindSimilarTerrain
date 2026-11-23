"""Benchmark: same 3 km x 3 km, 2 m DTM at native and coarser search resolutions.
Run: python tests/benchmark.py [n_cells]"""
import math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import terrain_core as tc, synth

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
cs_list = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [20, 10, 5, 2]
W = n * 2.0
d = synth.make_dem(n=n, ref_center=(400.0, 400.0),
                   targets=((W - 500, W - 450, 73.0, 100.0, 1.0),),
                   decoys=((500.0, W - 500, 200.0, -40.0, 2.0),),
                   hidden=((W - 450, 450.0, 20.0, 50.0, 1.0),))
geo = d["geo"]; dem = tc.ArrayDEM(d["z"], geo)
rw = synth.world_rings(d["ref_local"], 400, 400, 0.0)
srch = synth.search_rings_for((0, 0, W, W), (W - 450, 450.0), exclude_ref=(400, 400, 230))
print("DEM %d x %d cells at 2 m (%.1f km); reference ~%.0f m radius, with hole" % (n, n, W / 1000, 95))
print("%-26s %8s %9s %8s %9s %8s %9s" % ("config", "time_s", "pos_err_m", "ang_err", "rmse_m", "off_m", "recovered"))
for cs in cs_list:
    for refine in ([False] if cs == 2 else [True]):
        t0 = time.time()
        res = tc.find_similar_terrain(dem, rw, srch, comparison_cs=cs, refine_native=refine)
        el = time.time() - t0
        a = (res.anchor_xy[0] - 400, res.anchor_xy[1] - 400)
        t = d["truth"]["targets"][0]
        tx, ty = synth.true_anchor(a, t[0], t[1], t[2])
        m = res.matches[0]
        pe = math.hypot(m.x - tx, m.y - ty); ae = abs((m.angle - 73 + 180) % 360 - 180)
        ok = pe < 5 and ae < 1
        print("%-26s %8.1f %9.2f %8.3f %9.4f %8.2f %9s   (angles=%d, scan %.1fs, final cs %g)" % (
            "search %gm%s" % (cs, " + native refine" if refine else ""), el, pe, ae, m.rmse, m.offset,
            "YES" if ok else "NO", res.n_angles, res.timings["scan"], res.final_cs), flush=True)
