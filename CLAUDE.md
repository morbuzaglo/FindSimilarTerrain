# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An ArcGIS Pro 3.5 Python toolbox (`FindSimilarTerrain.pyt`) that finds the N locations in a search polygon whose terrain best matches a reference polygon (rotation 0-360 deg, score = RMSE of height differences with the mean offset reported separately, no normalisation). Docs are in Hebrew (`README_HE.md`, `FindSimilarTerrain_UserGuide_HE.docx`) and English (`README.md`).

## Commands

Everything needs the ArcGIS Pro Python (NumPy 1.26, SciPy 1.14, arcpy); there is no build or lint step.

```
PY="C:/Program Files/ArcGIS/Pro/bin/Python/envs/arcgispro-py3/python.exe"
$PY tests/test_core.py            # 16 pure-numpy tests (no arcpy needed)
$PY tests/test_arcgis.py          # end-to-end: loads the real .pyt via arcpy.ImportToolbox
$PY tests/benchmark.py 1500 20,10,5,2   # size_in_cells, search cell sizes
```

Tests are plain scripts printing PASS/FAIL (no pytest); to run one check, edit/call the function in the script directly.

## Architecture

- `terrain_core.py`: pure NumPy/SciPy engine, **no arcpy**. Entry point `find_similar_terrain(dem, ref_rings, search_rings, ...)` takes a DEM object (`ArrayDEM` in tests, `ArcDEM` in the toolbox) and polygon rings, returns `SearchResult` with `Match` objects. Pipeline: aggregate DTM to the comparison cell (area-weighted sparse matrices, coverage >= 0.5) -> rasterise polygons to masks (even-odd, holes work) -> masked-FFT correlation scores every placement per angle (`fft_rmse_scan`, tiled with halo) -> optional pyramid (`pyramid_levels`/`pyramid_seeds`, factor 2.5, only active when reference radius > ~12.5 cells) -> seeds via NMS -> pattern-search refinement in (x, y, theta) (`refine_pose`) -> optional refinement on the native-resolution DTM -> dedupe by overlap.
- `FindSimilarTerrain.pyt`: ArcGIS glue only: `ArcDEM` (windowed `RasterToNumPyArray` reads), polygon read/project (`read_polygon_rings`, honours layer selections, projects to the DTM CRS), parameter validation (auto cell-size suggestion), output feature class, optional residual raster, and `make_rank_symbology` (builds a `.lyrx`: unique colour per Rank, 50% transparency, assigned to `parameters[5].symbology`). It imports `terrain_core` from the same folder and `importlib.reload`s it, so both files must stay together.
- Parameter order in the `.pyt` is load-bearing (`p[0]`..`p[14]` indexes in `execute`/validation); Advanced group = 6-14. Update `FindSimilarTerrain.FindSimilarTerrain.pyt.xml` (help text) and the READMEs when changing parameters.
- `tests/synth.py` builds synthetic terrain with a known planted site; both test scripts depend on it.

## Conventions and gotchas

- Rotation: counter-clockwise viewed from above, about the centroid of the reference mask cells; pure rotation + translation (no scale/mirror). Keep this consistent across code, field `Angle_deg`, and docs.
- The DTM must be a local, projected (metric), square-cell raster. Geographic rasters and online image services (e.g. Terrain3D) are unsupported; the engine guards with `MAX_SEARCH_CELLS = 60e6`.
- Heights are scored in metres (`mpu` converts DTM units); output fields: Rank, RMSE_m, Offset_m, Angle_deg, Cell_m, SearchCell_m, CoarseRMSE_m, Mode.
- The `.pyt` uses `arcpy.management.CreateFeatureclass` (lowercase c).
- Not tracked (gitignored): `out/`, `data_cop/`, `*.tif`, `*.gdb/`, auto-generated `FindSimilarTerrain.pyt.xml`, and personal scripts in `tests/` (`_patch_pyramid.py`, `probe_service.py`, `make5m.py`, `prep_srtm.py`, `ref_stats.py`, `run_myproject.py`, `test_validator.py`).
- `docs/figures/*.svg` are the manual's figures (generated outside the repo; the Word manual is built separately and uses the David font, never Arial). Commits in this public repo use the GitHub noreply author email.
