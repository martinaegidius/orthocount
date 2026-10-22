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
    category_mapping: dict[int, int]
    @property 
    def num_classes(self) -> int: 
        """Number of foreground classes selected for training.""" 
        return len(self.category_mapping)

def load_labels(filepath: Path, data_cfg: DictConfig) -> COCOData:
    logger.info(f"Loading labels from {filepath}...")
    with open(filepath) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = defaultdict(list)
    
    categories = {
            cat["id"]: cat["name"]
            for cat in coco["categories"] if cat["id"] in data_cfg.use_classes
        }

    #need to remap category labels to contiguous N for the model. 0: background, 1: next class, ... cannot have 1,3 
    category_mapping = { 
        coco_id: model_id for model_id, coco_id in enumerate( 
            categories.keys(), start=1, 
            ) 
        }

    
    for ann in coco["annotations"]:
        if ann["category_id"] in data_cfg.use_classes: #keep as is - conversion using remapping will be done in torch dataclass
            annotations[ann["image_id"]].append(ann)
        


    num_annotations = sum(
        len(anns)
        for anns in annotations.values()
    )

    logger.info(
        f"Loaded labels for {len(images)} image files. "
        f"{len(annotations)} images contain labels. "
        f"{num_annotations} selected instances across "
        f"{len(categories)} classes."
    )

    logger.info(f"Categories: {categories}")
    logger.info(f"Category mapping: {category_mapping}")

    return COCOData(
        images=images,
        annotations=annotations,
        categories=categories,
        category_mapping=category_mapping,
    )
    

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

def load_manifest(cfg: DictConfig) -> dict:
    """
    Loads the tile manifest allowing mapping from tile filename and to crs coordinates.

    Args:
        cfg (DictConfig): hydra main configuration

    Returns:
        dict: a jsonl-type dictionary
    """
    data_dir = Path(cfg.paths.output_tiff).parent.parent 
    manifest_path = data_dir / "metadata" / "tiles_manifest.jsonl" 
    with open(manifest_path, "r") as f:
         metadata_list = [json.loads(line) for line in f]
    #convert list to dictionary where each key is the image filename
    metadata_dict = {}
    for elem in metadata_list:
        metadata_dict.update({elem["image"]:{}})
        metadata_dict[elem["image"]].update({"data":elem})

    return metadata_dict

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")   
def main(cfg: DictConfig) -> None: 
    filepath = cfg.paths.labels_filepath
    cocoData = load_labels(filepath)
    print(cocoData.images[1])
    print(cocoData.annotations[cocoData.images[1]["id"]])
    print(cocoData.categories)
    #debugging plot 
    debugging_plot(cocoData, Path(cfg.paths.tile_dir))

    
if __name__=="__main__":
    main()