from pathlib import Path
import json 
import copy

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger 
import torch 
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import v2 as T
from torchvision.io import read_image, ImageReadMode
from torchvision import tv_tensors


from orthocount.data.coco import load_labels, COCOData, load_manifest
from orthocount.models.helpers import load_model_from_checkpoint
from orthocount.evaluation.visualizations import visualize_prediction
from orthocount.predict.postprocessing import postprocess_prediction, global_nms, export_vector_shape_file

from tqdm import tqdm 

def collate_fn(batch):
    return tuple(zip(*batch))


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")   
def predict(cfg: DictConfig):
    logger.info("Running inference based on coco_data set.")
    torch.manual_seed(0) #same as train - to get same split 
    
    output_path = Path("predictions")
    output_path.mkdir(exist_ok=True)
    logger.info(f"Saving to {output_path}")
    filepath = cfg.paths.labels_filepath
    label_data = load_labels(filepath, cfg.training.labels)
    metadata_manifest = load_manifest(cfg)
    class_dict = label_data.categories

    model = load_model_from_checkpoint(cfg)
    predictions = run_prediction(cfg,model, label_data, class_dict, metadata_manifest, output_root=output_path)    
    if cfg.prediction.cvat_preannotation:
        export_cvat_pre_annotation(cfg, predictions, output_path)
    merged_predictions = global_nms(cfg, predictions)
    export_vector_shape_file(merged_predictions, output_path)


def export_cvat_pre_annotation(cfg: DictConfig, result_dict: dict, output_root: Path) -> None: 
    """
    Transforms predictions into coco-style pre-annotations importable to CVAT.

    Args:
        cfg (DictConfig): hydra main configuration
        result_dict (dict): the predictions from run_prediction
    """
    logger.info("Generating pre-annotations...")
    src_coco_filepath = cfg.paths.labels_filepath
    with open(src_coco_filepath) as f:
        src_preannotation_dict = json.load(f)
    target_preannotation_dict = copy.deepcopy(src_preannotation_dict)#
    all_annotated_im_ids = set([x["image_id"] for x in src_preannotation_dict["annotations"]])
    #all_im_ids = set([x["id"] for x in src_preannotation_dict["images"]])
    annotation_id = max(
        (ann["id"] for ann in src_preannotation_dict["annotations"]),
        default=0,
    ) + 1
    for im in tqdm(src_preannotation_dict["images"]):
        filename = im["file_name"] #this is the key of the prediction dict output from the model
        im_id = im["id"]
        if im_id in all_annotated_im_ids:
            continue  #never touch pre-annotated files as they are still a better ground truth than model annotations
        prediction = result_dict[filename]["prediction"][0]
        for box, score, label in zip(prediction["boxes"],prediction["scores"],prediction["labels"]):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            target_preannotation_dict["annotations"].append({
                "id": annotation_id,
                "image_id": im_id,
                "category_id": label,
                "bbox": [
                    x1,
                    y1,
                    width,
                    height,
                ],
                "segmentation": [],
                "area": width * height,
                "iscrowd": 0,
                "attributes": {"occluded":False,"rotation":0.0, "preannotated": True, "model_score": float(score),}
            })
            annotation_id += 1
    out_file = output_root / "coco_preannotations.json"
    with open(out_file, "w") as f:
        json.dump(target_preannotation_dict, f, indent=2)
    logger.info(f"Saved CVAT pre-annotations to {out_file}")
    return 



def run_prediction(cfg: DictConfig,model,label_data: COCOData, class_dict: dict, metadata_dict: dict, output_root: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.model
    model = model.to(device)
    
    model.eval()
    tile_path = Path(cfg.paths.tile_dir)
    result_dict = {}
    if cfg.prediction.visualize_predictions:
        im_out_path_root = output_root / "visualizations"
        im_out_path_root.mkdir(exist_ok=True)
    with torch.no_grad():
        for entry_id, value in tqdm(label_data.images.items()):   #just like a dataloader

            image_filepath = tile_path / value["file_name"] #remember to add full filepath
            image = read_image(image_filepath, mode=ImageReadMode.RGB) #C x H x W     
            image = image.float() / 255.0
            
            image = tv_tensors.Image(image).to(device)
            prediction = model([image])
            prediction = postprocess_prediction(cfg,prediction) #nms etc

            prediction_json = [
                {
                    "boxes": p["boxes"].cpu().tolist(),
                    "scores": p["scores"].cpu().tolist(),
                    "labels": p["labels"].cpu().tolist(),
                }
                for p in prediction
            ]
            result_dict[value["file_name"]] = value
            result_dict[value["file_name"]].update({"prediction":prediction_json,
                                                    "metadata":metadata_dict[value["file_name"]]})
            if cfg.prediction.visualize_predictions:
                im_out_path = im_out_path_root / f"{value["file_name"]}" #maybe consider original filename instead? this makes programming cleaner, though
                visualize_prediction(image=image, #first image in batch - i.e. when batch_size is 1 it is always the first image! type: list
                                        prediction=prediction[0],
                                        out_fname=im_out_path,
                                        class_names = class_dict) #TODO: support batches. not relevant on my shoddy GPU which can only handle a single image anyways :)))
    out_file = output_root / "predictions_tiled_nms.json"
    with open(out_file, "w") as f:
        json.dump(result_dict, f, indent=2)
    logger.info(f"Saved result dict prior to global NMS to {out_file}")
    logger.info("Finished tiled predictions!")
    return result_dict



if __name__=="__main__":
    predict()
    
#TODO: also make config. Missing currently.<3 

    
