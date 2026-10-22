from pathlib import Path   
import torch 
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class COCOImageDataset(Dataset):
    def __init__(self, image_dir, label_data, transform=None):
        #self.img_dir = img_dir
        self.image_dir = Path(image_dir)
        self.images = label_data.images 
        #self.image_ids = list(self.images.keys()) #would include "non-labelled" images aswell
        self.image_ids = list(label_data.annotations.keys())
        self.annotations = label_data.annotations
        self.category_mapping = label_data.category_mapping
        self.transform = transform
        
    def __len__(self):
        return len(self.image_ids)     

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image_filename = self.image_dir / image_info["file_name"]
        metadata = self.annotations.get(image_id, [])
        boxes = []
        class_ids = []
        areas = []
        for ann in metadata:
            x, y, w, h = ann["bbox"]

            boxes.append([
                x,
                y,
                x + w,
                y + h
            ]) #format: x1,y1,x2,y2 - expected by torchvision

            #class_ids.append(ann["category_id"])
            class_ids.append(self.category_mapping[ann["category_id"]]) #remapping - model needs class ids to be contiguous and non-zero as zero is background
            areas.append(ann["area"])


        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        ).reshape(-1, 4)

        class_ids = torch.tensor(
            class_ids,
            dtype=torch.int64
        )

        areas = torch.tensor(
                    areas,
                    dtype=torch.float32
                )
        image = read_image(image_filename, mode=ImageReadMode.RGB) #C x H x W     
        image = image.float() / 255.0

        image = tv_tensors.Image(image)

        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image))
        iscrowd = torch.zeros(len(areas),dtype=torch.uint8) #all false
        target = {
            "boxes": boxes,
            "labels": class_ids, #are class_ids, as expected by pretrained models
            "image_id": image_id,
            "area": areas, 
            "iscrowd": iscrowd, 
            "all_metadata": metadata
        }

        if self.transform is not None:
            image, target = self.transform(image, target) #note: currently area will probably not be handled by transforms. Will be important for evaluations downstream if we decide to keep. 
            #todo: also apply augmentation to area field to ensure correct area. 
        return image, target