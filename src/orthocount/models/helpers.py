from pathlib import Path 

from omegaconf import DictConfig

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from loguru import logger 

from orthocount.models.lighting_module import PavillionDetector 

def load_model(cfg: DictConfig):
    """
    Simply a parser for loading models :)))

    Args:
        cfg (DictConfig): Hydra root config

    Raises:
        NotImplementedError: Error for model types not implemented yet.
    """
    if cfg.training.model.lower()=="fasterrcnn":
        model = fasterrcnn_mobilenet_v3_large_fpn(
                    weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
                    progress=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes=len(cfg.training.labels.use_classes)+1,   # background and white pavilion for starters
        )
        
    else:
        raise NotImplementedError
    return model 

def load_model_from_checkpoint(cfg: DictConfig, model_path = None): 
    detector = load_model(cfg)
    if model_path is None: 
        logger.info("Auto-detecting model checkpoint...")
        path = Path("checkpoints/")
        checkpoint_path = list(path.glob("*.ckpt"))
        if len(checkpoint_path)>1:
            logger.error("multiple checkpoints detected. Specify which to load.")
        elif len(checkpoint_path)==0:
            raise FileNotFoundError("No model checkpoint found!")
        else: 
            model_path = checkpoint_path[0]
    if cfg.training.model.lower()=="fasterrcnn":
        model = PavillionDetector.load_from_checkpoint(model_path, model=detector)
        logger.info("Loaded Faster R-CNN pavilion detector.")

    else:
        raise NotImplementedError(
            f"Unknown model: {cfg.training.model}"
        )

    
    return model 
    