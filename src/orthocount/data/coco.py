import json
from collections import defaultdict
from dataclasses import dataclass 
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger 

import matplotlib.pyplot as plt 
from matplotlib import patches

@dataclass
class COCOData:
    images: dict[int, dict]
    annotations: dict[int, list[dict]]
    categories: dict[int, str]
    
def load_labels(filepath: Path) -> COCOData:
    logger.info(f"Loading labels from {filepath}...")
    with open(filepath) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    annotations = defaultdict(list)
    
    categories = {
            cat["id"]: cat["name"]
            for cat in coco["categories"]
        }

    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)
    
    num_annotations = len(coco["annotations"])
    num_annotated_files = len(annotations)    
    logger.info(f"Loaded labels for {len(images)} image files. {num_annotated_files} images contain labels. {num_annotations} instances across IDs in total.")
    return COCOData(images=images, annotations=dict(annotations),categories=categories)

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")   
def main(cfg: DictConfig) -> None: 
    filepath = cfg.paths.labels_filepath
    cocoData = load_labels(filepath)
    print(cocoData.images[1])
    print(cocoData.annotations[cocoData.images[1]["id"]])
    print(cocoData.categories)
    #debugging plot 
    debugging_plot(cocoData, Path(cfg.paths.tile_dir))


def debugging_plot(label_set: COCOData, im_root: Path, idx: int = 1) -> None: 
    
    image = label_set.images[idx]
    labels = label_set.annotations[image["id"]]
    #print(label_set.categories)
    colordict = {1:"red", 2:"cyan",3:"orange",4:"yellow"}
    bboxes = [x["bbox"] for x in labels]
    class_ids = [x["category_id"] for x in labels]
    im = plt.imread(im_root / image["file_name"])
    fig, ax = plt.subplots(1,1)
    ax.imshow(im)
    
    for box, cat in zip(bboxes, class_ids): 
        xmin, ymin, w, h = box
        box_patch = patches.Rectangle(
            (xmin, ymin), w, h, edgecolor=colordict[cat], facecolor="none"
        )
        ax.add_patch(box_patch)
    plt.show()

    
if __name__=="__main__":
    main()