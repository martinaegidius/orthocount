from pathlib import Path
import json 

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger 
import torch 
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchmetrics.detection import MeanAveragePrecision

from orthocount.data.coco import load_labels
from orthocount.data.dataset import COCOImageDataset
from orthocount.models.helpers import load_model_from_checkpoint
from orthocount.evaluation.visualizations import visualize_prediction
from orthocount.predict.postprocessing import postprocess_prediction


def collate_fn(batch):
    return tuple(zip(*batch))


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")   
def evaluate(cfg: DictConfig):
    logger.info("Running evaluation.")
    torch.manual_seed(0) #same as train - to get same split 
    output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "evaluation"
    output_path.mkdir(exist_ok=True)
    logger.info(f"Saving to {output_path}")
    filepath = cfg.paths.labels_filepath
    label_data = load_labels(filepath, cfg.training.labels)
    class_dict = label_data.categories
    dataset = COCOImageDataset(image_dir=Path(cfg.paths.tile_dir), label_data = label_data)
    N_samples = len(dataset)
    model = load_model_from_checkpoint(cfg)
    val_loader = None
    train_loader = None
    test_loader = None
    

    indices = torch.randperm(N_samples).tolist()
                
    if not cfg.training.training.overfit:
        frac_train = 0.8  #should later on be inherited from cfg.
        N_train = int(frac_train * len(indices))
        train_indices = indices[:N_train]
        val_indices = indices[N_train:]
        val_set = Subset(dataset,val_indices)
        val_loader = DataLoader(val_set, batch_size=1,shuffle=False, collate_fn=collate_fn)

        if cfg.evaluation.evaluate_train: 
            train_set = Subset(dataset, train_indices)
            train_loader = DataLoader(train_set, batch_size=1,shuffle=False, collate_fn=collate_fn)
    else:
        logger.info("Running overfit experiment! train = val")
        N_train = cfg.training.training.N_overfit_samples
        train_indices = indices[:N_train]
        train_set = Subset(dataset, train_indices)                
        train_loader = DataLoader(train_set, batch_size=1,shuffle=False, collate_fn=collate_fn)
    
    result_dict = run_evaluation(cfg,model,train_loader,val_loader,test_loader, class_dict, output_path)    
    logger.info(f"Evaluation results: \n\t{result_dict}")

    summary_output_path = output_path / "mAP_results.json"
    with open(summary_output_path, "w") as f:
        json.dump(result_dict,f)
    logger.info(f"Saved all metrics to {summary_output_path}")

    return result_dict

def run_evaluation(cfg: DictConfig,model,train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,test_loader: torch.utils.data.DataLoader, class_dict: dict, output_root: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strings = ["train", "validation", "test"]
    loaders = [train_loader,val_loader,test_loader]
    model = model.model
    model = model.to(device)
    model.eval()
    result_dict = {}
    data_manifest = load_manifest(cfg)

    with torch.no_grad():
        for str_desc, dataloader in zip(strings,loaders):
            
            sample_results = []
            metric = MeanAveragePrecision(box_format="xyxy",iou_type='bbox',class_metrics=True)
                                
            if dataloader is None: 
                continue
            else:    
                result_dict.update({f"{str_desc}_samples":[]})
                result_dict.update({f"{str_desc}_map":{}})
                split_root_path = Path(str_desc)
                split_root_path.mkdir(exist_ok=True)
                im_out_path_root = output_root / split_root_path
                im_out_path_root.mkdir(exist_ok=True)
                            
                logger.info(f"Evaluating: {str_desc}. Outputs will be saved to: {split_root_path}")
            for i, batch in enumerate(dataloader):
                images, target = batch 
                target = [
                    {
                        k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in t.items()
                    }
                    for t in target
                ]
                images = tuple(img.to(device) for img in images)
                prediction = model(images)
                metric.update(prediction,target) 

                #gather per-sample evaluation metrics: 
                """
                sample_result = evaluate_detection(
                                         prediction=prediction[0],
                                         target=target[0],
                                         iou_threshold=0.5,
                                         confidence_threshold=0.5
                                         class_names = class_dict)
                
                """
                #breakpoint()
                #sample_result = (target[0]["image_id"],2,2,2,2) #TODO
                #sample_results.append(sample_result)
                            
                if cfg.evaluation.visualize_evaluation_preds:
                    im_out_path = im_out_path_root / f"{i}.png"
                    prediction = postprocess_prediction(cfg,prediction)

                    visualize_prediction(image=images[0].cpu(), #first image in batch - i.e. when batch_size is 1 it is always the first image! type: list
                                         prediction=prediction[0],
                                         out_fname=im_out_path,
                                         target=target[0],
                                         threshold=0.5,
                                         class_names = class_dict) #TODO: support batches. not relevant on my shoddy GPU which can only handle a single image anyways :)))

            logger.info("\tCalculating mAP values...")
            metrics = metric.compute()
            metrics_cpu = {k:v.cpu().item() if v.dim()==0 else v.tolist() for k,v in metrics.items()}
            result_dict[f"{str_desc}_map"].update(metrics_cpu)
            result_dict.update({f"{str_desc}_samples":sample_results})
                           

    return result_dict

if __name__=="__main__":
    #evaluation_results_output_path = Path("evaluation_outputs")
    #logger.info(f"Saving results to {evaluation_results_output_path}")
    #evaluation_results_output_path.mkdir(exist_ok=True)

    result_dict = evaluate()



    
