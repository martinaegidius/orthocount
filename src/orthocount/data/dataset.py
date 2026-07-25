from pathlib import Path   
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

class COCOImageDataset(Dataset):
    def __init__(self, image_dir, images, annotations, transform=None, target_transform=None):
        #self.img_dir = img_dir
        self.image_dir = Path(image_dir)
        self.images = images 
        self.image_ids = list(images.keys())
        self.annotations = annotations
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_ids)     

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image_filename = self.image_dir / image_info["file_name"]
        labels = self.annotations.get(image_id, [])
        bboxes = [x["bbox"] for x in labels]
        class_ids = [x["category_id"] for x in labels]
        image = read_image(image_filename, mode=ImageReadMode.RGB) #C x H x W         
        target = {
            "boxes": bboxes,
            "class_ids": class_ids,
            "labels": labels,
            "image_id": image_id
        }
        return image, target