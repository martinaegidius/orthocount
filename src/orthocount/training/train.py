import math 
from pathlib import Path


import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger 
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint 

import torch  
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import v2 as T

#from orthocount.models.detector import LitAutoEncoder

from orthocount.data.coco import load_labels
from orthocount.data.dataset import COCOImageDataset
from orthocount.models.lighting_module import PavillionDetector
from orthocount.models.helpers import load_model
import orthocount.evaluation.evaluation_metrics


def collate_fn(batch):
    return tuple(zip(*batch))

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="pavilion-{epoch:02d}-{val_map_50:.3f}",
    monitor="val_map_50",
    mode="max",
    save_top_k=1,
)

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")   
def train(cfg: DictConfig):
    torch.manual_seed(0)
    filepath = cfg.paths.labels_filepath
    label_data = load_labels(filepath, cfg.training.labels)
    train_transforms = T.Compose([
        T.RandomRotation(180),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),

        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.02,
        ),
         T.SanitizeBoundingBoxes(),
    ])
    train_dataset = COCOImageDataset(image_dir=Path(cfg.paths.tile_dir), label_data = label_data, transform=train_transforms)
    val_dataset = COCOImageDataset(image_dir=Path(cfg.paths.tile_dir), label_data = label_data, transform=None) #keeping unaugmented as validation
    N_samples = len(train_dataset)
    indices = torch.randperm(N_samples).tolist()
            
    if not cfg.training.training.overfit:
        frac_train = cfg.training.training.frac_train
        N_train = int(frac_train * len(indices))
        train_indices = indices[:N_train]
        val_indices = indices[N_train:]
#        train_set, val_set = random_split(dataset, [N_train, N_samples-N_train])
    else:
        logger.info("Running overfit experiment! train = val")
        N_train = cfg.training.training.N_overfit_samples
        train_indices = indices[:N_train]
        val_indices = indices[:N_train]
        
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_set, batch_size=cfg.training.training.batch_size,shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=cfg.training.training.batch_size,shuffle=False, collate_fn=collate_fn)
    #image, target = next(iter(train_loader))
    
    #print(target.keys()) # list of entries. dict_keys(['boxes', 'class_ids', 'labels', 'image_id']). Len = num objects. 
    logger.info(f"Dataloader has: {len(train_loader)} samples")
    model = load_model(cfg)
    
    #loss_dict = model(image, target)
    #loss = sum(loss_dict.values())
    
    # ---------------------------------------------------------------
    # Lightning wrapper
    # ---------------------------------------------------------------

    lightning_model = PavillionDetector(
        model=model,
        learning_rate=cfg.training.training.learning_rate,
    )

    # ---------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------
    accumulate_grad_batch_size = cfg.training.training.accumulate_grad_batch_size
    trainer = L.Trainer(
        max_epochs=cfg.training.training.max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        accumulate_grad_batches=accumulate_grad_batch_size,
        check_val_every_n_epoch=cfg.training.training.val_every,
        callbacks=[checkpoint_callback] #save checkpoints, please
    )

    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == "__main__":
    train()
