import json 
from pathlib import Path
import time 


import geopandas as gpd 
import hydra
from loguru import logger 
import numpy as np 
from omegaconf import DictConfig, OmegaConf
from osgeo import gdal 
from PIL import Image 
import rasterio 
from rasterio import windows 
from shapely.geometry import box 



#from torch.utils.data import Dataset

#def downsample(cfg: dict):

class RFTifOrthophoto():
    def __init__(self, geotiff_path: Path, vector_path: Path, im_size: int, overlap: int) -> None:
        self.tiff_path = Path(geotiff_path) #../data/2025-07-02-Roskilde-Festival_20cm.tif
        self.vector_path = Path(vector_path)  #../data/camping_zones.shp
        self.data_dir = self.tiff_path.parent.parent 
        self.im_size = im_size 
        self.overlap = overlap 
        self.stride = im_size-overlap
    

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder =  self.data_dir 
        im_out_dir = self.data_dir / "tiles"
        metadata_output_dir = output_folder / "metadata"
        manifest_path = metadata_output_dir / "tiles_manifest.jsonl"
        im_out_dir.mkdir(parents=True, exist_ok=True)
        metadata_output_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(self.tiff_path) as src:
            with open(manifest_path, "a") as manifest_file:
                raster_crs = src.crs 
                gdf = gpd.read_file(self.vector_path)
                gdf = gdf.to_crs(raster_crs)
                merged_zones = gdf.dissolve(by="Zone key") #some camping zones are disjoint in annotation 
                """
                Tiling - consider making a standalone function for this later. 
                """
                for i, (zone_pol, zone_key) in enumerate(zip(merged_zones["geometry"],merged_zones.index)):
                    minx, miny, maxx, maxy = zone_pol.bounds #left, bottom, right, top
                    row_min, col_min = src.index(minx, maxy) #convert coords to raster indices
                    row_max, col_max = src.index(maxx, miny)
                    file_counter = 0
                    for row in range(row_min, row_max, self.stride): #read from upper left to lower right, row-major order
                        for col in range(col_min, col_max, self.stride):
                            window = windows.Window(col, row, self.im_size, self.im_size) #pixel-land
                            left, bottom, right, top = windows.bounds(window,src.transform) #in coordinates
                            tile_geom = box(left, bottom, right, top)
                            if not tile_geom.intersects(zone_pol):
                                continue
                            
                            tile = src.read(window=window)
                            tile = tile.transpose(1,2,0) #RGBA but in 4 x h x w format 
                            tile_id = f"zone_{zone_key}_{file_counter:05d}"
                            tile_filename = f"{tile_id}.png"
                            tile_out = im_out_dir / tile_filename
                            Image.fromarray(tile).convert("RGB").save(tile_out)
                            file_metadata = {
                                "tile_id": tile_id,
                                "image": tile_filename,
                                "zone_key": zone_key,
                                "row_off": row,
                                "col_off": col,
                                "width": self.im_size,
                                "height": self.im_size,
                                "bounds": [left, bottom, right, top],
                                "transform": list(src.window_transform(window))
                            }
                            manifest_file.write(json.dumps(file_metadata) + "\n")
                            file_counter+=1
                    
@hydra.main(version_base=None, config_path="../../configs", config_name="config")   
def preprocess(cfg: DictConfig) -> None:
    """
    Is run directly together with the downsample pipeline (or at least be called with the corresponding configuration which uses the gdal_preprocessing configuration file for determining output_tiff)

    Args:
        cfg (DictConfig): hydra main config
    """
    logger.info("Tiling...")
    dataset = RFTifOrthophoto(geotiff_path=cfg.paths.output_tiff, vector_path=cfg.paths.shape_filepath,im_size=cfg.tiling.im_size,overlap=cfg.tiling.overlap)
    dataset.preprocess()

@hydra.main(version_base=None, config_path="../../configs", config_name="config")   
def gdal_preprocess(cfg: DictConfig) -> None:
    sampling_cfg = cfg.gdal_preprocessing 
    input_tiff = cfg.paths.src_tiff
    output_dir = Path(cfg.paths.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gdal.UseExceptions()
    with gdal.Open(input_tiff) as ds:
        #info = gdal.Info(ds, format='json')
        #del info["stac"]  # to avoid cluttering below output
        #logger.info(info.keys())
        gt = ds.GetGeoTransform()
    
        pixel_res_x = gt[1]
        pixel_res_y = gt[5]
    
        logger.info(f"Pixel Resolution X: {pixel_res_x}") 
        logger.info(f"Pixel Resolution Y: {pixel_res_y}") 
    
        assert np.isclose(np.abs(pixel_res_x),np.abs(pixel_res_y)), "Error - pixel pitch is unequal. Check tiff. Aborting procedures"
    target_resolution = sampling_cfg.physical_pixel_resolution
    #output_tiff = output_dir / f"output_{target_resolution}m_cog.tif" <- moved to configs 
    logger.info(cfg)
    logger.info(cfg.paths.src_tiff)
    logger.info(cfg.paths.processed_dir)
    logger.info(output_dir)
    gdal_convert(str(input_tiff), str(cfg.paths.output_tiff), target_resolution, cfg=cfg.gdal_preprocessing)



def gdal_convert(input_path: str, output_path: str, target_res: float, cfg):
    gdal.UseExceptions()
    
    creation_options = [
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "NUM_THREADS=ALL_CPUS",
        "BIGTIFF=YES"
    ]
    if not cfg.cog:
        creation_options.extend(["TILED=YES", #in cog case is tiled by default
                                f"BLOCKXSIZE={cfg.BLOCKSIZE}",
                                f"BLOCKYSIZE={cfg.BLOCKSIZE}"])
        fmt = "GTiff"
    else:
        fmt = "COG"
        creation_options.extend([
            f"BLOCKSIZE={cfg.BLOCKSIZE}",
            "RESAMPLING=AVERAGE",
            "OVERVIEWS=IGNORE_EXISTING"]
        )
    # Configure the Warp options
    warp_options = gdal.WarpOptions(
        format=fmt,                    # -of COG
        xRes=target_res,                 # -tr x_res
        yRes=target_res,                 # -tr y_res (GDAL handles the negative sign implicitly here)
        resampleAlg=gdal.GRIORA_Average, # -r average
        creationOptions=creation_options
    )
    
    logger.info(f"Warping {input_path} to {output_path} at {target_res}m resolution...")
    
    start = time.time()
    gdal.Warp(output_path, input_path, options=warp_options)
    end = time.time()
    
    logger.info(f"Processing complete. Saved downsampled output to {output_path}.\nElapsed time: {end-start}")


if __name__ == "__main__":
    gdal_preprocess()
    preprocess()


