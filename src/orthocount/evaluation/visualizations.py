from pathlib import Path 

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_prediction(
    image,
    prediction,
    out_fname: Path,
    target=None,
    threshold=0.5,
    class_names=None,
):
    """
    Visualize model predictions and optionally ground-truth boxes.

    Args:
        image: Image tensor [C, H, W].
        prediction: Faster R-CNN prediction dictionary.
        out_fname: Output image path.
        target: Optional ground-truth target dictionary.
        threshold: Minimum prediction confidence to visualize.
        class_names: Mapping from class ID to class name, e.g.
            {
                1: "white pavilion",
                2: "rented pavilion",
                3: "other pavilion",
            }
    """

    if class_names is None:
        class_names = {}

    image = image.cpu().permute(1, 2, 0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    # ---------------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------------

    if target is not None:
        gt_boxes = target["boxes"].cpu()
        gt_labels = target["labels"].cpu()

        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            label_id = int(label)

            class_name = class_names.get(
                label_id,
                f"class {label_id}",
            )

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2,
                color="cyan",
                label="Ground truth",
            )

            ax.add_patch(rect)

            ax.text(
                x1,
                y1,
                f"GT: {class_name}",
                fontsize=8,
                color="cyan",
                verticalalignment="bottom",
            )

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------

    pred_boxes = prediction["boxes"].cpu()
    scores = prediction["scores"].cpu()
    pred_labels = prediction["labels"].cpu()

    for box, score, label in zip(
        pred_boxes,
        scores,
        pred_labels,
    ):
        if score < threshold:
            continue

        x1, y1, x2, y2 = box.tolist()
        label_id = int(label)

        class_name = class_names.get(
            label_id,
            f"class {label_id}",
        )

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
            color="red",
            label="Prediction",
        )

        ax.add_patch(rect)

        ax.text(
            x1,
            y1,
            f"Pred: {class_name} ({score:.2f})",
            fontsize=8,
            color="red",
            verticalalignment="top",
        )

    # Avoid duplicate legend entries when there are many boxes
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    ax.axis("off")
    fig.tight_layout()

    plt.savefig(out_fname, bbox_inches="tight")
    plt.close(fig)


"""
def visualize_prediction(
    image,
    prediction,
    out_fname: Path,
    target=None,
    threshold=0.5,
):
    
    Visualize model predictions and optionally ground-truth boxes.

    Predictions are drawn in one style and ground-truth boxes in another,
    making it easy to identify false positives and missed detections.
    
    image = image.cpu().permute(1, 2, 0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    # ---------------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------------

    if target is not None:
        gt_boxes = target["boxes"].cpu()

        for box in gt_boxes:
            x1, y1, x2, y2 = box.tolist()

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2,
                color="cyan",
                label="Ground truth",
            )

            ax.add_patch(rect)

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------
    
    pred_boxes = prediction["boxes"].cpu()
    scores = prediction["scores"].cpu()

    for box, score in zip(pred_boxes, scores):

        if score < threshold:
            continue

        x1, y1, x2, y2 = box.tolist()

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
            label="Prediction",
            color="red"
        )

        ax.add_patch(rect)

        ax.text(
            x1,
            y1,
            f"{score:.2f}",
            fontsize=8,
            verticalalignment="bottom",
        )

    # Avoid duplicate legend entries when there are many boxes
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    ax.axis("off")
    fig.tight_layout()

    plt.savefig(out_fname, bbox_inches="tight")
    plt.close(fig)

"""