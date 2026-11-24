# -*- coding: utf-8 -*-
"""Find Similar Terrain - ArcGIS Pro 3.5 Python toolbox.

GIS input/output via ArcPy; all numerics in terrain_core.py (NumPy/SciPy).
Keep terrain_core.py in the same folder as this .pyt.
"""
import importlib
import math
import os
import sys
import time

import arcpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrain_core as tc  # noqa: E402
importlib.reload(tc)       # pick up edits without restarting Pro


# ---------------------------------------------------------------------------
# ArcPy helpers
# ---------------------------------------------------------------------------
class ArcDEM:
    """Windowed reader with the interface terrain_core expects."""

    def __init__(self, raster_path):
        self.ras = arcpy.Raster(raster_path)
        r = self.ras
        if abs(r.meanCellWidth - r.meanCellHeight) > 1e-6 * r.meanCellWidth:
            raise ValueError("The DTM must have square cells.")
        self.sr = r.spatialReference
        if self.sr is None or not self.sr.type == "Projected":
            raise ValueError("The DTM must be in a projected coordinate system (metric distances "
                             "are needed). Project it first.")
        self.mpu = self.sr.metersPerUnit or 1.0
        self.geo = tc.Grid(r.extent.XMin, r.extent.YMax, r.meanCellWidth, r.height, r.width)
        self._float = str(r.pixelType).upper().startswith("F")
        self._nd = r.noDataValue
        self._arr = tc.ArrayDEM(None, self.geo)   # reuse window maths

    def window(self, xmin, ymin, xmax, ymax):
        g = self.geo
        c0 = max(int(math.floor((xmin - g.x0) / g.cs + 1e-9)), 0)
        c1 = min(int(math.ceil((xmax - g.x0) / g.cs - 1e-9)), g.ncols)
        r0 = max(int(math.floor((g.ytop - ymax) / g.cs + 1e-9)), 0)
        r1 = min(int(math.ceil((g.ytop - ymin) / g.cs - 1e-9)), g.nrows)
        if c1 <= c0 or r1 <= r0:
            return None, None
        ll = arcpy.Point(g.x0 + c0 * g.cs, g.ytop - r1 * g.cs)
        if self._float:
            a = arcpy.RasterToNumPyArray(self.ras, ll, c1 - c0, r1 - r0, nodata_to_value=np.nan)
            a = np.asarray(a, dtype=np.float64)
        else:
            nd = self._nd if self._nd is not None else -32768
            a = arcpy.RasterToNumPyArray(self.ras, ll, c1 - c0, r1 - r0, nodata_to_value=nd)
            a = np.asarray(a, dtype=np.float64)
            a[a == nd] = np.nan
        a[~np.isfinite(a)] = np.nan
        a[a < -1e30] = np.nan
        return a, tc.Grid(g.x0 + c0 * g.cs, g.ytop - r0 * g.cs, g.cs, r1 - r0, c1 - c0)


def _layer_text(param):
    v = param.valueAsText
    return v.strip("'") if v else v


def read_polygon_rings(source, target_sr, max_dev):
    """Union of the (selected) polygons of `source` -> list of (N,2) rings
    in target_sr (outer rings and holes together).  Returns (rings, n_feat,
    used_selection)."""
    desc = arcpy.Describe(source)
    used_sel = False
    try:
        fid = desc.FIDSet
        used_sel = bool(fid)
    except Exception:
        pass
    geoms = []
    with arcpy.da.SearchCursor(source, ["SHAPE@"]) as cur:
        for (g,) in cur:
            if g is None:
                continue
            geoms.append(g)
    if not geoms:
        raise ValueError("No polygon features in '%s' (or no features selected)." % source)
    g = geoms[0]
    for h in geoms[1:]:
        g = g.union(h)
    if g.spatialReference and target_sr and g.spatialReference.factoryCode != target_sr.factoryCode:
        g = g.projectAs(target_sr)
    try:
        if g.hasCurves:
            g = g.densify("DISTANCE", max_dev * 5, max_dev)
    except Exception:
        pass
    rings = []
    for part in g:
        cur_ring = []
        for p in part:
            if p is None:
                if len(cur_ring) >= 3:
                    rings.append(np.array(cur_ring, float))
                cur_ring = []
            else:
                cur_ring.append((p.X, p.Y))
        if len(cur_ring) >= 3:
            rings.append(np.array(cur_ring, float))
    return rings, len(geoms), used_sel


def rings_to_polygon(rings, sr):
    arrs = arcpy.Array()
    for r in rings:
        arrs.add(arcpy.Array([arcpy.Point(float(x), float(y)) for x, y in r]))
    return arcpy.Polygon(arrs, sr)


def default_output():
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        gdb = aprx.defaultGeodatabase
    except Exception:
        gdb = arcpy.env.workspace or arcpy.env.scratchGDB
    return arcpy.CreateUniqueName("SimilarTerrain", gdb)


def _sum_area(source, sr):
    """(total area m2, bbox area m2) of the polygons, projected to `sr` first
    (inputs may be geographic).  Honours layer selections."""
    mpu = (sr.metersPerUnit or 1.0) if sr else 1.0
    tot = 0.0
    ext = None
    with arcpy.da.SearchCursor(source, ["SHAPE@"]) as cur:
        for (shp,) in cur:
            if shp is None:
                continue
            if sr is not None and shp.spatialReference and shp.spatialReference.factoryCode != sr.factoryCode:
                shp = shp.projectAs(sr)
            tot += shp.area
            e = shp.extent
            ext = e if ext is None else ext.union(e)
    if ext is None:
        raise ValueError("no polygons")
    return tot * mpu ** 2, ext.width * ext.height * mpu ** 2


# ---------------------------------------------------------------------------
class Toolbox(object):
    def __init__(self):
        self.label = "Find Similar Terrain"
        self.alias = "FindSimilarTerrain"
        self.tools = [FindSimilarTerrain]


RANK_COLORS = [(230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48), (145, 30, 180),
               (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40),
               (128, 0, 0), (0, 0, 128)]


def make_rank_symbology(fc, transparency=50, aprx=None):
    """Build a .lyrx that colours every polygon by Rank (unique colour each, listed by Rank)
    at ~50% transparency. Returns the .lyrx path, or None if no ArcGIS Pro project is open."""
    try:
        if aprx is None:
            aprx = arcpy.mp.ArcGISProject("CURRENT")
        mp = aprx.activeMap or aprx.listMaps()[0]
        lyr = mp.addDataFromPath(fc)
        try:
            sym = lyr.symbology
            sym.updateRenderer("UniqueValueRenderer")
            sym.renderer.fields = ["Rank"]
            for item in sym.renderer.groups[0].items:
                try:
                    k = int(float(item.values[0][0]))
                except Exception:
                    continue
                r, g, b = RANK_COLORS[(k - 1) % len(RANK_COLORS)]
                item.symbol.color = {"RGB": [r, g, b, 100]}
                item.symbol.outlineColor = {"RGB": [r, g, b, 100]}
                item.symbol.size = 1.5
                item.label = "Rank %d" % k
            lyr.symbology = sym
            lyr.transparency = transparency
            lyr.name = os.path.basename(fc)
            out = os.path.join(arcpy.env.scratchFolder, "similar_terrain_ranks.lyrx")
            lyr.saveACopy(out)
        finally:
            mp.removeLayer(lyr)
        return out
    except Exception as e:  # not running inside Pro, or symbology unavailable
        arcpy.AddMessage("Layer symbology not applied (%s)." % e)
        return None



class FindSimilarTerrain(object):
    def __init__(self):
        self.label = "Find Similar Terrain"
        self.description = (
            "Finds the best-matching locations in a search area for the terrain inside a reference "
            "polygon. Candidates are scored with the actual elevations: d = candidate - reference, "
            "RMSE = sqrt(mean((d - mean(d))^2)) in meters; mean(d) is reported as the elevation "
            "offset. Rotation about the vertical axis (0-360 deg) is searched by default.")
        self.canRunInBackground = False
        self.category = "Terrain Analysis"

    # ------------------------------------------------------------------
    def getParameterInfo(self):
        p_dem = arcpy.Parameter("dem", "DTM (elevation raster)", "Input", "GPRasterLayer", "Required")
        p_ref = arcpy.Parameter("reference_polygon", "Reference polygon", "Input", "GPFeatureLayer", "Required")
        p_ref.filter.list = ["Polygon"]
        p_srch = arcpy.Parameter("search_polygon", "Search polygon", "Input", "GPFeatureLayer", "Required")
        p_srch.filter.list = ["Polygon"]
        p_cs = arcpy.Parameter("cell_size", "Comparison cell size (meters)", "Input", "GPDouble", "Optional")
        p_cs.filter.type = "Range"
        p_cs.filter.list = [0.01, 100000.0]
        p_ref_nat = arcpy.Parameter("refine_native", "Refine best matches at native resolution",
                                    "Input", "GPBoolean", "Optional")
        p_ref_nat.value = True
        p_out = arcpy.Parameter("out_fc", "Output matches", "Output", "DEFeatureClass", "Required")

        adv = "Advanced"
        p_n = arcpy.Parameter("n_results", "Number of results", "Input", "GPLong", "Optional")
        p_n.value = 5
        p_n.filter.type = "Range"
        p_n.filter.list = [1, 100]
        p_rot = arcpy.Parameter("rotate", "Search all rotations (0-360 degrees)", "Input", "GPBoolean", "Optional")
        p_rot.value = True
        p_step = arcpy.Parameter("coarse_step", "Coarse angle step (degrees; blank = automatic)",
                                 "Input", "GPDouble", "Optional")
        p_step.filter.type = "Range"
        p_step.filter.list = [0.1, 90.0]
        p_exh = arcpy.Parameter("exhaustive", "Exhaustive search (verification, small areas)",
                                "Input", "GPBoolean", "Optional")
        p_exh.value = False
        p_estep = arcpy.Parameter("exhaustive_step", "Exhaustive angle increment (degrees; blank = automatic)",
                                  "Input", "GPDouble", "Optional")
        p_estep.filter.type = "Range"
        p_estep.filter.list = [0.05, 45.0]
        p_ovl = arcpy.Parameter("max_overlap", "Maximum overlap between results (%)", "Input", "GPDouble", "Optional")
        p_ovl.value = 20.0
        p_ovl.filter.type = "Range"
        p_ovl.filter.list = [0.0, 100.0]
        p_sl = arcpy.Parameter("shortlist", "Shortlist size for refinement (blank = automatic)",
                               "Input", "GPLong", "Optional")
        p_res = arcpy.Parameter("out_residual", "Output signed residual raster (best match)",
                                "Output", "DERasterDataset", "Optional")
        p_pyr = arcpy.Parameter("pyramid", "Multi-resolution pyramid search (fast)", "Input", "GPBoolean", "Optional")
        p_pyr.value = True
        for p in (p_n, p_rot, p_step, p_exh, p_estep, p_ovl, p_sl, p_res, p_pyr):
            p.category = adv
        return [p_dem, p_ref, p_srch, p_cs, p_ref_nat, p_out,
                p_n, p_rot, p_step, p_exh, p_estep, p_ovl, p_sl, p_res, p_pyr]

    def isLicensed(self):
        return True

    # ------------------------------------------------------------------
    def updateParameters(self, parameters):
        p = parameters
        if not p[5].altered and not p[5].value:
            try:
                p[5].value = default_output()
            except Exception:
                pass
        # auto-suggest comparison cell size until the user types one
        changed = any(q.altered and not q.hasBeenValidated for q in p[:3])
        if changed and not p[3].altered and p[0].value and p[1].value and p[2].value:
            try:
                dem_path = arcpy.Describe(_layer_text(p[0])).catalogPath
                r = arcpy.Raster(dem_path)
                mpu = r.spatialReference.metersPerUnit or 1.0
                ncs = r.meanCellWidth * mpu
                sr = r.spatialReference
                a_ref, _ = _sum_area(_layer_text(p[1]), sr)
                _, a_srch = _sum_area(_layer_text(p[2]), sr)
                p[3].value = round(tc.suggest_cell_size(ncs, a_ref, a_srch), 4)
            except Exception:
                pass
        if p[7].value is False:
            p[8].enabled = False
        else:
            p[8].enabled = True
        p[10].enabled = bool(p[9].value)

    def updateMessages(self, parameters):
        p = parameters
        if p[3].value is not None and p[0].value:
            try:
                r = arcpy.Raster(arcpy.Describe(_layer_text(p[0])).catalogPath)
                sr = r.spatialReference
                if sr.type != "Projected":
                    p[0].setErrorMessage("The DTM must be in a projected coordinate system.")
                    return
                ncs = r.meanCellWidth * (sr.metersPerUnit or 1.0)
                cs = p[3].value
                if cs < ncs * 0.999:
                    p[3].setWarningMessage("Native DTM cell size is %g m; a finer comparison cell "
                                           "size is not possible and native will be used." % ncs)
                elif p[1].value:
                    a_ref, _ = _sum_area(_layer_text(p[1]), sr)
                    r_cells = math.sqrt(a_ref / math.pi) / cs
                    if r_cells < 6:
                        p[3].setWarningMessage(
                            "Native DTM cell size is %g m. At %g m the reference polygon is only "
                            "~%.0f cells in radius; narrow parts will not be captured. Use a smaller "
                            "cell size." % (ncs, cs, r_cells))
            except Exception:
                pass
        if p[9].value and p[7].value is not False:
            p[9].setWarningMessage("Exhaustive mode is slow: it evaluates every grid placement at a "
                                   "fine angular increment. Use it on small areas for verification.")

    # ------------------------------------------------------------------
    def execute(self, parameters, messages):
        T0 = time.time()
        p = parameters
        add = arcpy.AddMessage
        dem_path = arcpy.Describe(_layer_text(p[0])).catalogPath
        dem = ArcDEM(dem_path)
        mpu = dem.mpu
        sr = dem.sr
        ncs_m = dem.geo.cs * mpu
        add("DTM: %s  |  native cell size %g m  |  %d x %d cells" %
            (os.path.basename(dem_path), ncs_m, dem.geo.ncols, dem.geo.nrows))

        cs_m = p[3].value
        cs_units = None if cs_m is None else float(cs_m) / mpu
        dev = dem.geo.cs / 4.0
        ref_rings, nref, sel1 = read_polygon_rings(_layer_text(p[1]), sr, dev)
        srch_rings, nsrch, sel2 = read_polygon_rings(_layer_text(p[2]), sr, dev)
        add("Reference polygon: %d feature(s)%s; search polygon: %d feature(s)%s." %
            (nref, " (selected)" if sel1 else "", nsrch, " (selected)" if sel2 else ""))

        n_res = int(p[6].value or 5)
        rotate = p[7].value is not False
        exhaustive = bool(p[9].value)

        arcpy.SetProgressor("step", "Starting...", 0, 100, 1)

        def progress(frac, msg=None):
            if arcpy.env.isCancelled if hasattr(arcpy.env, "isCancelled") else False:
                raise RuntimeError("Cancelled by user.")
            arcpy.SetProgressorPosition(int(100 * frac))
            if msg:
                arcpy.SetProgressorLabel("%s  [%.0fs]" % (msg, time.time() - T0))

        res = tc.find_similar_terrain(
            dem, ref_rings, srch_rings,
            comparison_cs=cs_units,
            refine_native=bool(p[4].value) if p[4].value is not None else True,
            n_results=n_res, rotate=rotate, exhaustive=exhaustive,
            coarse_step=p[8].value or None, exhaustive_step=p[10].value or None,
            shortlist=p[12].value or None,
            pyramid=(p[14].value is not False),
            max_overlap=(p[11].value if p[11].value is not None else 20.0) / 100.0,
            progress=progress, log=add)

        for n in res.notes:
            add(n)
        for w in res.warnings:
            arcpy.AddWarning(w)
        # exact containment check with ArcPy geometry (handles curves/holes)
        srch_geom = rings_to_polygon(srch_rings, sr)
        good = []
        for m in res.matches:
            poly = rings_to_polygon(m.rings, sr)
            if srch_geom.contains(poly):
                good.append((m, poly))
            else:
                arcpy.AddWarning("Candidate at (%.1f, %.1f) was not entirely inside the search "
                                 "polygon (ArcPy check) and was discarded." % (m.x, m.y))
        for i, (m, _) in enumerate(good, 1):
            m.rank = i

        # ---- output feature class ---------------------------------------
        out = p[5].valueAsText
        out_path, out_name = os.path.split(out)
        if not out_path:
            out_path = arcpy.env.workspace
        arcpy.management.CreateFeatureclass(out_path, out_name, "POLYGON", spatial_reference=sr)
        fc = os.path.join(out_path, out_name)
        for name, typ in (("Rank", "LONG"), ("RMSE_m", "DOUBLE"), ("Offset_m", "DOUBLE"),
                          ("Angle_deg", "DOUBLE"), ("Cell_m", "DOUBLE"), ("SearchCell_m", "DOUBLE"),
                          ("CoarseRMSE_m", "DOUBLE")):
            arcpy.management.AddField(fc, name, typ)
        arcpy.management.AddField(fc, "Mode", "TEXT", field_length=255)
        with arcpy.da.InsertCursor(fc, ["SHAPE@", "Rank", "RMSE_m", "Offset_m", "Angle_deg", "Cell_m",
                                        "SearchCell_m", "CoarseRMSE_m", "Mode"]) as cur:
            for m, poly in good:
                cur.insertRow([poly, m.rank, m.rmse, m.offset, m.angle, m.final_cs * mpu,
                               m.search_cs * mpu, m.coarse_rmse,
                               ("exhaustive" if exhaustive else "coarse-to-fine")[:255]])

        lyrx = make_rank_symbology(fc)
        if lyrx:
            p[5].symbology = lyrx

        # ---- optional residual raster for the best match -------------------
        if p[13].value and good:
            arr, g = tc.residual_grid(dem, ref_rings, res.anchor_xy, good[0][0])
            ras = arcpy.NumPyArrayToRaster(arr.astype(np.float32),
                                           arcpy.Point(g.x0, g.ymin), g.cs, g.cs, np.nan)
            ras.save(p[13].valueAsText)
            arcpy.management.DefineProjection(p[13].valueAsText, sr)
            add("Residual raster saved: (candidate - reference - offset) in meters, native cells "
                "inside the best-match polygon; positive = candidate higher than reference.")

        # ---- report -------------------------------------------------------
        add("-" * 60)
        add("Search resolution: %g m   |   final scoring resolution: %g m" %
            (res.search_cs * mpu, res.final_cs * mpu))
        add("Rotation search (angle = degrees counter-clockwise viewed from above, 0-360): " + res.mode)
        add("Coarse screening can miss terrain features smaller than the comparison cell size.")
        add("Rank   RMSE_m   Offset_m   Angle_deg   Anchor X / Y")
        for m, _ in good:
            add("%4d  %7.3f  %9.2f  %9.2f    %.2f, %.2f" % (m.rank, m.rmse, m.offset, m.angle, m.x, m.y))
        tm = res.timings
        add("Timing: read/aggregate %.1fs, scan %.1fs, refine %.1fs, total %.1fs" %
            (tm.get("read_aggregate", 0), tm.get("scan", 0),
             tm.get("refine_search_res", 0) + tm.get("refine_native", 0), time.time() - T0))
        arcpy.ResetProgressor()
        return

    def postExecute(self, parameters):
        return
