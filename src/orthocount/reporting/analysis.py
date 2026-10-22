from pathlib import Path
import json 

import hydra
from omegaconf import DictConfig, OmegaConf
import geopandas as gpd 
import rasterio 
from loguru import logger

import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

def load_prediction_dict(cfg: DictConfig) -> dict:
    pred_path = Path("predictions/predictions_global_nms.gpkg")
    preds = gpd.read_file(pred_path)
    preds = preds.astype({"label":"int8"})
    return preds
    

def run_analysis(cfg: DictConfig, merge_on: str) -> None:
    tiff_path = Path(cfg.paths.output_tiff) 
    zone_vector_path= Path(cfg.paths.shape_filepath)

    zone_count_dict = {}

    #data_dir = tiff_path.parent.parent 
    with rasterio.open(tiff_path) as src:
        raster_crs = src.crs 
        gdf = gpd.read_file(zone_vector_path)
        gdf = gdf.to_crs(raster_crs)

        merged_zones = gdf.dissolve(by=merge_on) #some camping zones are disjoint in annotation 
        merged_zones = merged_zones.reset_index()
        predictions = load_prediction_dict(cfg)
        if predictions.crs != merged_zones.crs:
            logger.info(f"Detected different CRS between predictions/shapefile:{predictions.crs}!={merged_zones.crs}. Auto-converting..")
            predictions = predictions.to_crs(merged_zones.crs)
        
        predictions_with_zone = gpd.sjoin(
            predictions,
            merged_zones[[merge_on, "geometry"]],
            how="left",
            predicate="intersects",
        )
        #get num of individual occurences per zone
        zone_count_dict = {}
        for zone in predictions_with_zone[merge_on].unique():
            zone_count_dict.update({zone: {}})
            all_labels = predictions_with_zone[predictions_with_zone[merge_on]==zone]
            n_white_pavillion = len(all_labels[all_labels["label"]==1])
            n_rented_pavillion = len(all_labels[all_labels["label"]==2])
            n_other_pavillion = len(all_labels[all_labels["label"]==3])
            zone_count_dict[zone].update({"white_pavillion":n_white_pavillion,"rented_pavillion":n_rented_pavillion,"other_pavillion":n_other_pavillion})
        
    count_path = Path("predictions") / "analysis"
    count_path.mkdir(exist_ok=True)

    output_json = count_path / f"{merge_on}_counts.json"

    with open(output_json,"w") as f:
        json.dump(zone_count_dict,f)
    logger.info(f"Saved output to {output_json}")

    output_shapefile = count_path / f"{merge_on}_count_geography.gpkg"
    predictions_with_zone. #TODO
    return zone_count_dict

def export_barchart(zc: dict, merged_by: str) -> None: 
    fig, ax = plt.subplots(2,1,figsize=(8,12))
    df = (
        pd.DataFrame(zc)
        .T
        .rename_axis(merged_by.lower())
        .reset_index()
    )

    df = df.melt(
        id_vars=merged_by.lower(),
        var_name="pavillion_type",
        value_name="count"
    )
    df = df.sort_values("count",ascending=False)

    sns.barplot(
        data=df,
        x=merged_by.lower(),
        y="count",
        hue="pavillion_type",ax=ax[0]
    )
        
    #plt.savefig("raw_counts.pdf",dpi=300)
    #plt.close()
        # Normalize counts within each zone
    df["proportion"] = (
        df["count"]
        / df.groupby(merged_by.lower())["count"].transform("sum")
    )
    #df = df.sort_values(by="proportion",ascending=False)
    
    white_order = (
        df.loc[(df["pavillion_type"] == "white_pavillion")& df[merged_by.lower()].notna()]
        .sort_values("proportion", ascending=False)[merged_by.lower()]
    )   
    
    sns.barplot(
        data=df,
        x=merged_by.lower(),
        y="proportion",
        hue="pavillion_type",
        ax=ax[1],
        order = white_order
    )

    ax[1].set_ylabel("Proportion")
    
    ax[0].set_xticks(ax[0].get_xticks(), ax[0].get_xticklabels(), rotation=45, ha='right')
    ax[1].set_xticks(ax[1].get_xticks(), ax[1].get_xticklabels(), rotation=45, ha='right')
        


    plt.tight_layout()
    plt.savefig(f"{merged_by}_results.pdf",dpi=300)
    #todo: zone keys need to be mapped to actual zone names...

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")   
def main(cfg: DictConfig):
    merge_on = ["Zone key", "Type"]
    for merge_key in merge_on:
        zone_counts = run_analysis(cfg,merge_on=merge_key)
        export_barchart(zone_counts, merged_by=merge_key)

    
if __name__=="__main__":
    main()