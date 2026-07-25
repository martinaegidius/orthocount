from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger 
import lightning as L
from torch import nn 

from models.model import LitAutoEncoder

from data.coco import load_labels
from data.dataset import COCOImageDataset

from torch.utils.data import DataLoader 

@hydra.main(version_base=None, config_path="../../configs", config_name="config")   
def train(cfg: DictConfig):
    filepath = cfg.paths.labels_filepath
    label_data = load_labels(filepath)
    dataset = COCOImageDataset(image_dir=Path(cfg.paths.tile_dir), images=label_data.images, annotations=label_data.annotations)
    train_loader = DataLoader(dataset)
    logger.info(f"Dataloader has: {len(train_loader)} samples")
    ##TODO: MAKE SPLITS AND REMEMBER THAT UNLABELLED FILES SHOULD BE IN INFERENCE SPLIT 
    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    # define any number of nn.Modules (or use your current ones)
    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
    # init the autoencoder
    autoencoder = LitAutoEncoder(encoder, decoder)

    trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)
    

if __name__ == "__main__":
    train()
