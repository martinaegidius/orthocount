import json 
from pathlib import Path

import geopandas as gpd 
import numpy as np 
from PIL import Image 
from rasterio import windows 
from shapely.geometry import box 
from torch.utils.data import Dataset
import typer


class RFOrthophoto(Dataset):
    def __init__(self, geotiff_path: Path, vector_path: Path, im_size: int, overlap: int) -> None:
        self.tiff_path = geotiff_path #../data/2025-07-02-Roskilde-Festival_20cm.tif
        self.vector_path = vector_path  #../data/camping_zones.shp
        self.data_dir = self.tiff_path.parent 
        self.im_size = im_size 
        self.overlap = overlap 
        self.stride = im_size-overlap
    

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        im_out_dir = output_folder / "tiles"
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
                    if i == 0: #currently not production - only a single zone for rapid prototyping 
                        break
   
def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = RFOrthophoto(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)




