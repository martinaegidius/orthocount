from osgeo import gdal

for path in [
    "./data/2025/canonical/geotiff/0.1m_cog_False.tif",
    "./data/2026/input_resolution/geotiff/RF2026_GSD10cm.tif",
]:
    ds = gdal.Open(path)

    print(f"\n{path}")
    print("Projection:", ds.GetProjection())
    print("GeoTransform:", ds.GetGeoTransform(can_return_null=True))
    print("Raster size:", ds.RasterXSize, ds.RasterYSize)
    print("Metadata:", ds.GetMetadata())
