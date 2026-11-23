# Find Similar Terrain (ArcGIS Pro 3.5 Python toolbox)

**[מדריך בעברית ← README_HE.md](README_HE.md)** (user manual and algorithm explanation in Hebrew; Word version: [FindSimilarTerrain_UserGuide_HE.docx](FindSimilarTerrain_UserGuide_HE.docx))

Files (keep together): `FindSimilarTerrain.pyt`, `terrain_core.py` (NumPy/SciPy engine),
`FindSimilarTerrain.FindSimilarTerrain.pyt.xml` (tool help). `tests/` holds tests and the benchmark.

## Run in ArcGIS Pro
1. Catalog pane > Toolboxes > right-click > Add Toolbox > `FindSimilarTerrain.pyt`.
2. Open **Find Similar Terrain**. Choose the DTM (projected CRS, square cells), a reference polygon layer,
   a search polygon layer. If a layer has a selection only the selected polygons are used.
   (To "draw": digitise a polygon in any polygon feature class, or select an existing one.)
3. **Comparison cell size (meters)** is auto-suggested (about 10 cells of reference radius); change it freely
   (e.g. 5, 10, 20 on a 2 m DTM). The DTM's native size is printed in the tool messages and a warning appears
   if the cell size is finer than native or too coarse for the polygon.
4. Leave **Refine best matches at native resolution** on. Output goes to the project geodatabase (`SimilarTerrain`).
5. By default a multi-resolution pyramid search is used (coarse screening of the whole area, then local re-scoring of surviving candidates at each finer level): about 10-50x faster on large areas. Turn it off under Advanced to scan every placement at the comparison cell size.
6. Advanced: result count (5), rotations (all), coarse angle step, exhaustive mode, max overlap, shortlist, residual raster.

Output fields: `Rank, RMSE_m, Offset_m (mean of candidate - reference), Angle_deg, Cell_m (final scoring cell), SearchCell_m, CoarseRMSE_m`.
Angle: degrees counter-clockwise viewed from above (0-360), reference rotated about its cell centroid then moved.
Score: `d = candidate - reference`, `RMSE = sqrt(mean((d-mean(d))^2))`; nothing (relief, slope) is normalised.

Caveats: coarse screening can miss terrain features smaller than the comparison cell size. Exhaustive mode is exact only
on the stated grid and angle increment (printed in the messages); results are then refined continuously.
Vertical units are assumed to be meters.

DTM requirements: local raster, projected (metric) CRS, square cells. Geographic rasters and online image services (e.g. Terrain3D) must first be projected/clipped with Project Raster (set the processing extent).

## Tests (use ArcGIS Pro's Python: `C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe`)
    python tests\test_core.py       # numerics, brute-force comparison, synthetic known-site, NoData, holes
    python tests\test_arcgis.py     # runs the real .pyt via arcpy.ImportToolbox on a GeoTIFF + gdb polygons
    python tests\benchmark.py 1500 20,10,5,2
