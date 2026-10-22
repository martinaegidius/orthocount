import lightning as L
import torch
from torch.optim import AdamW
from torchmetrics.detection import MeanAveragePrecision


#from orthocount.models.detector import fasterRNN - will be added if we need to actually change stuff. 

class PavillionDetector(L.LightningModule):
    def __init__(self, model, learning_rate: float = 1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.val_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=True,
        )
        
    def forward(self, images, targets: dict = None):
        return self.model(images,targets)

    def training_step(self, batch: tuple, batch_idx: int):
        images, targets = batch
        loss_dict = self.model(images,targets)
        loss = sum(loss_dict.values()) #calculated internally in the torchvision model
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size = len(images)
        )

        for name, value in loss_dict.items():
            self.log(
                f"train/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=len(images)
            )
        
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        images, targets = batch
        self.model.train()# Faster R-CNN returns losses only while in training mode. Move towards other metrics later.

        with torch.no_grad():
            loss_dict = self.model(images, targets)

        loss = sum(loss_dict.values())
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(images),
        )
        #also evaluate predictions
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(images)

        self.val_map.update(
            predictions,
            targets,
        )
        
        # Lightning will put the module back into the
        # appropriate mode when necessary.

    def on_validation_epoch_end(self):

        metrics = self.val_map.compute()

        self.log("val_map_50",metrics["map_50"],prog_bar=True)
        self.log("val_map_50_95",metrics["map"])
        self.log("val_mar_100",metrics["mar_100"])
        self.val_map.reset()


    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        return optimizer
