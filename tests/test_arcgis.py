"""End-to-end test through ArcGIS Pro's own toolbox loader.
Run with the Pro python:  propy tests/test_arcgis.py
Builds a 2 m GeoTIFF with NoData, reference/search polygons (with holes,
a SELECTION on the layer) in a scratch file gdb, runs the .pyt, checks results."""
import math, os, shutil, sys, time
import numpy as np
import arcpy

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
sys.path.insert(0, root); sys.path.insert(0, here)
import synth, terrain_core as tc

work = os.path.join(root, "out", "arc_test")
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
arcpy.env.overwriteOutput = True
gdb = arcpy.management.CreateFileGDB(work, "t.gdb").getOutput(0)
sr = arcpy.SpatialReference(32636)   # UTM 36N, meters
n = 800; ox, oy = 500000.0, 3500000.0
d = synth.make_dem(n=n, ref_center=(250., 250.), targets=((1100., 1150., 73., 100., 1.),),
                   decoys=((400., 1200., 200., -40., 2.),), hidden=((1200., 350., 20., 50., 1.),))
z = d["z"].astype(np.float32); z[300:340, 700:760] = np.nan      # NoData patch
tif = os.path.join(work, "dtm.tif")
r = arcpy.NumPyArrayToRaster(z, arcpy.Point(ox, oy), 2.0, 2.0, np.nan)
r.save(tif); arcpy.management.DefineProjection(tif, sr)

def poly_fc(name, rings_local):
    fc = arcpy.management.CreateFeatureclass(gdb, name, "POLYGON", spatial_reference=sr).getOutput(0)
    with arcpy.da.InsertCursor(fc, ["SHAPE@"]) as c:
        arr = arcpy.Array([arcpy.Array([arcpy.Point(ox + x, oy + y) for x, y in rg]) for rg in rings_local])
        c.insertRow([arcpy.Polygon(arr, sr)])
        # a decoy extra feature that must NOT be used when it is not selected
        c.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Array([arcpy.Point(ox+10, oy+10), arcpy.Point(ox+10, oy+60), arcpy.Point(ox+60, oy+60), arcpy.Point(ox+60, oy+10)])]), sr)])
    return fc
refl = synth.world_rings(d["ref_local"], 250, 250, 0.0)
ref_fc = poly_fc("ref", refl)
srch = synth.search_rings_for((0, 0, n*2, n*2), (1200, 350), exclude_ref=(250, 250, 230))
srch_fc = poly_fc("srch", srch)
reflyr = arcpy.management.MakeFeatureLayer(ref_fc, "reflyr", "OBJECTID = 1").getOutput(0)  # layer definition
srchlyr = arcpy.management.MakeFeatureLayer(srch_fc, "srchlyr")
arcpy.management.SelectLayerByAttribute("srchlyr", "NEW_SELECTION", "OBJECTID = 1")  # select only the big polygon
tb = arcpy.ImportToolbox(os.path.join(root, "FindSimilarTerrain.pyt"))
out = os.path.join(gdb, "matches"); resid = os.path.join(work, "resid.tif")
t0 = time.time()
arcpy.FindSimilarTerrain_FindSimilarTerrain(tif, "reflyr", "srchlyr", 10, True, out, 5, True, None, False, None, 20, None, resid)
el = time.time() - t0
print(arcpy.GetMessages())
rows = sorted(arcpy.da.SearchCursor(out, ["Rank", "RMSE_m", "Offset_m", "Angle_deg", "Cell_m", "SearchCell_m", "SHAPE@"]))
print("ELAPSED %.1fs" % el)
fails = []
def chk(name, ok, det=""):
    print(("PASS " if ok else "FAIL ") + name + (" | " + det if det else "")); 
    if not ok: fails.append(name)
chk("5 ranked polygons written", len(rows) == 5, str(len(rows)))
b = rows[0]
c = b[6].centroid
# true centroid: polygon is the reference rotated 73 deg + translated: compare using centroid of ref
rc = arcpy.Polygon(arcpy.Array([arcpy.Array([arcpy.Point(*p) for p in rg]) for rg in refl]), sr).centroid
a_loc = (rc.X - 250, rc.Y - 250)   # centroid offset (ref polygon centroid vs mask centroid differ slightly)
tx, ty = synth.true_anchor((rc.X-250, rc.Y-250), 1100., 1150., 73.)
dist = math.hypot(c.X - ox - tx, c.Y - oy - ty)
chk("best polygon centroid at the true site", dist < 3.0, "centroid error %.2f m" % dist)
chk("angle ~73", abs(b[3] - 73) < 0.5, "%.3f" % b[3])
chk("offset ~100", abs(b[2] - 100) < 0.5, "%.3f" % b[2])
chk("RMSE small, final cell 2 m, search cell 10 m", b[1] < 0.5 and abs(b[4] - 2) < 1e-6 and abs(b[5] - 10) < 1e-6, "rmse %.4f" % b[1])
sg = arcpy.Polygon(arcpy.Array([arcpy.Array([arcpy.Point(ox+x, oy+y) for x, y in rg]) for rg in srch]), sr)
chk("every result inside search polygon (arcpy contains, with holes)", all(sg.contains(r[6]) for r in rows))
chk("results exclude the unselected small reference decoy feature (ref area preserved)",
    all(abs(r[6].area - rows[0][6].area) / rows[0][6].area < 1e-6 for r in rows),
    "areas %.0f" % rows[0][6].area)
refarea = arcpy.Polygon(arcpy.Array([arcpy.Array([arcpy.Point(*p) for p in rg]) for rg in refl]), sr).area
chk("physical dimensions preserved (area equals reference)", abs(rows[0][6].area - refarea) / refarea < 1e-6, "%.1f vs %.1f" % (rows[0][6].area, refarea))
ras = arcpy.Raster(resid)
ra = arcpy.RasterToNumPyArray(ras, nodata_to_value=np.nan)
chk("residual raster written, ~0 residual for best match", np.nanmax(np.abs(ra)) < 1.0, "max |res| %.3f m" % np.nanmax(np.abs(ra)))
print("\nARC TESTS: %d failed" % len(fails)); sys.exit(1 if fails else 0)
