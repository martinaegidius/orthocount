import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt 
import geopandas as gpd 

gdf = gpd.read_file("data/camping_zones.shp")
print(gdf)



#with rasterio.open('data/2025-07-02-Roskilde-Festival_20cm.tif') as src:
    
    
#    w = src.read(1, window=Window(0, 0, 512, 256))
    #print(w.shape)
    #plt.imshow(w)
    #plt.show()
    