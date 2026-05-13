import os 

import rasterio 

def read_windowed():
    return 


def read_rbg(dataset: )
if __name__ == "__main__":
    data_dir = "geotiff_sample/sample.tif"
    dataset = rasterio.open(data_dir)
    print(dataset.keys())
    H, W, C = dataset.meta["height"], dataset.meta["width"], dataset.meta["count"]
    transform = dataset.meta["transform"] 
    print(dataset.bounds)
    dataset.transform*(0,0) #upper left corner
    
    


