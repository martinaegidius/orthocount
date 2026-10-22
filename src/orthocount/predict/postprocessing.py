from pathlib import Path 

import torch 
from torchvision.ops import nms 
from omegaconf import DictConfig
import numpy as np 
from affine import Affine
import geopandas as gpd
from shapely.geometry import box
from loguru import logger 


def postprocess_prediction(cfg: DictConfig, prediction: list[dict]) -> list[dict]:
    """
    Applies configuration based filters on a prediction.
    Prediction taken as list - assumes if batch_size = n -> len(prediction)=n

    Args:
        cfg (DictConfig): the hydra configuration
        prediction (list[dict]): model predictions from eval mode

    Returns:
        list[dict]: each field containing "boxes", "scores" and "labels" fields. 

    """
    if cfg.evaluation.filtering.confidence_filter.apply:
        prediction = confidence_filter_prediction(prediction,confidence_threshold=cfg.evaluation.filtering.confidence_filter.confidence_threshold)
    if cfg.evaluation.filtering.area_filter.apply:
        prediction = area_filter_prediction(prediction,min_area=cfg.evaluation.filtering.area_filter.min_area,max_area=cfg.evaluation.filtering.area_filter.max_area)                                        
    if cfg.evaluation.filtering.nms.apply:
        prediction = nms_filter_prediction(prediction,iou_threshold=cfg.evaluation.filtering.nms.iou_threshold)
    return prediction 

def nms_filter_prediction(prediction: list[dict],iou_threshold: float = 0.5) -> list[dict]:
    """
    Applies NMS on a single tile. Prediction taken as list - assumes if batch_size = n -> len(prediction)=n.
    Note: we assume detections to be exclusive, i.e. we simply assign the label associated with the max confidence class instance. 
    This should work fine in an ortophoto of plane objects in same height which should in principle not overlap. 

    Args:
        prediction (list[dict]): _description_
        iou_threshold (float, optional): _description_. Defaults to 0.5.

    Returns:
        list[dict]: same format as input prediction list, just with nms applied
    """
    filtered_outputs = []
    for batch_entry in prediction: 
        boxes = batch_entry["boxes"]
        scores = batch_entry["scores"]
        labels = batch_entry["labels"]
        idx_to_keep = nms(boxes=boxes,scores=scores,iou_threshold=iou_threshold)
        filtered_outputs.append({"boxes":boxes[idx_to_keep],"labels":labels[idx_to_keep],"scores":scores[idx_to_keep]})
    return filtered_outputs


def confidence_filter_prediction(prediction: list[dict], confidence_threshold: float = 0.5) -> list[dict]:
    """
    Applies confidence filtering on a single tile. Prediction taken as list - assumes if batch_size = n -> len(prediction)=n.
    
    Args:
        prediction (list[dict]): _description_
        confidence_threshold (float, optional): _description_. Defaults to 0.5.

    Returns:
        list[dict]: same format as input prediction list, just with filtered detections
    """
    filtered_outputs = []
    for batch_entry in prediction: 
        keep_confidence = batch_entry["scores"]>=confidence_threshold
        boxes = batch_entry["boxes"][keep_confidence]
        scores = batch_entry["scores"][keep_confidence]
        labels = batch_entry["labels"][keep_confidence]
        filtered_outputs.append({"boxes":boxes,"labels":labels,"scores":scores})
    return filtered_outputs


def area_filter_prediction(prediction: list[dict], min_area: float, max_area: float = None) -> list[dict]:
    """
    Applies area filtering on a single tile. Prediction taken as list - assumes if batch_size = n -> len(prediction)=n.
        
    Args:
        prediction (list[dict]): _description_. Assumes xyxy boxes prediction format
        min_area (float): format: pixels
        max_area (float, optional): pixels. Defaults to None. if None, it is unbound.

    Returns:
        list[dict]: same format as input prediction list, just with filtered detections
    
    """
    filtered_outputs = []
    for batch_entry in prediction: 
        boxes = batch_entry["boxes"]
        widths = boxes[:,2]-boxes[:,0]
        heights = boxes[:,3]-boxes[:,1]
        areas = widths*heights
        if max_area is not None:
            idx_to_keep = (areas>=min_area) & (areas<=max_area)
        else:
            idx_to_keep = areas>=min_area
        boxes = boxes[idx_to_keep]
        scores = batch_entry["scores"][idx_to_keep]
        labels = batch_entry["labels"][idx_to_keep]
        filtered_outputs.append({"boxes":boxes,"labels":labels,"scores":scores})
    return filtered_outputs


def convert_boxes_to_crs(arr: np.array, transform: list) -> dict:
    
    fwd_trf = Affine(*transform)
    x1 = arr[:,0]
    y1 = arr[:,1]
    x2 = arr[:,2]
    y2 = arr[:,3]

    x1_crs, y1_crs = fwd_trf * (x1,y1) #note - this flips the order of coordinates because increasing y is negative pixel resolution. 
    x2_crs, y2_crs = fwd_trf * (x2,y2)
    #meaning we get: x1_crs < x2_crs, y1_crs > y2_crs because of negative pixel resolution in y
    return np.column_stack([
        np.minimum(x1_crs, x2_crs),
        np.minimum(y1_crs, y2_crs),
        np.maximum(x1_crs, x2_crs),
        np.maximum(y1_crs, y2_crs),
    ]) #respect XYXY format for torchvision.ops

def global_nms(cfg: DictConfig, predictions: dict) -> dict:
    all_boxes = []
    all_scores = []
    all_labels = []
    all_metadata = []

    for k, v in predictions.items():
        bboxes = v["prediction"][0]["boxes"] #format: xyxy, only valid for single batch
        transform = v["metadata"]["data"]["transform"] #9 values, a list
        np_arr = np.array(bboxes)
        np_arr = np_arr.reshape(len(bboxes),4)
        crs_coords = convert_boxes_to_crs(np_arr,transform)    
        all_boxes.append(torch.from_numpy(crs_coords))
        all_scores.append(torch.tensor(v["prediction"][0]["scores"]))
        all_labels.append(torch.tensor(v["prediction"][0]["labels"]))
        all_metadata.extend(
            [
                {
                    "filename": k,
                    "prediction_index": i,
                }
                for i in range(len(bboxes))
            ]
        )

    all_boxes = torch.cat(all_boxes).float()
    all_scores = torch.cat(all_scores).float()
    all_labels = torch.cat(all_labels)
    idx_to_keep = nms(boxes=all_boxes,scores=all_scores,iou_threshold=cfg.evaluation.filtering.nms.iou_threshold)
    
    return {"boxes":all_boxes[idx_to_keep],"labels":all_labels[idx_to_keep],"scores":all_scores[idx_to_keep],"metadata": [all_metadata[i] for i in idx_to_keep.tolist()],}

def export_vector_shape_file(merged_dict: dict, output_path: Path) -> None:
    output_path = output_path / "predictions_global_nms.gpkg"
    

    boxes = merged_dict["boxes"].numpy()

    geometries = [
    box(xmin, ymin, xmax, ymax)
    for xmin, ymin, xmax, ymax in boxes
    ]

    gdf = gpd.GeoDataFrame(
    {
        "label": merged_dict["labels"].numpy(),
        "score": merged_dict["scores"].numpy(),
        "filename": [
            m["filename"]
            for m in merged_dict["metadata"]
        ],
        "prediction_idx": [
            m["prediction_index"]
            for m in merged_dict["metadata"]
        ],
    },
    geometry=geometries,
    crs="EPSG:25832",  # <-- use your actual CRS!
    )
    
    gdf.to_file(
        output_path,
        layer="pavilions",
        driver="GPKG",
    )

    logger.info(f"Saved output file to {output_path}")