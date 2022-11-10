import torch
from typing import List
from collections import Counter
from . import intersection_over_union


def mean_average_precision(
        boxes_model: List[List[float]],
        boxes_true: List[List[float]],
        box_format: str = "corners",
        iou_threshold: float = 0.5,
        num_classes: int = 1,
):
    """
    Mean average precision metric for n-class object detection
    Parameters:
        boxes_model (List): Box coordinates of model predictions. Shape of batch_size x 7.
            [image_idx, class_prediction, probability_score, x1, y1, x2, y2]
        boxes_true (List): Box coordinates of true data. Shape of batch_size x 7.
            [image_idx, class_prediction, probability_score, x1, y1, x2, y2]
        box_format (str): On which format bounding boxes are passed.
         Corner points or middle point with height and width. (x1, y1, x2, y2) in case of corners and
         (x_center, y_center, width, height) in case of midpoint.
        iou_threshold (float): Intersection over union threshold
        num_classes (int): Number of unique classes for objects
    Returns:
        mAP (float): Mean Average Precision for specified iou threshold
    """
    # Initialize list where elements = precision-recall auc for a specific class
    average_precisions = []
    for c in range(num_classes):
        # Container for predicted bounding boxes of class c
        detections = list(filter(lambda x: x[1] == c, boxes_model))
        # Container for ground true bounding boxes of class c
        ground_truths = list(filter(lambda x: x[1] == c, boxes_true))
        # Dictionary of the image ID and the number of its bounding boxes in the dataset
        bboxes_amount = Counter(list(map(lambda x: x[0], ground_truths)))
        # Converting number of bounding boxes from int to zeros tensor of this int len
        bboxes_amount = {k: torch.zeros(v) for k, v in bboxes_amount.items()}

        # Sort detections inplace by probability score, descending
        detections.sort(key=lambda x: x[2], reverse=True)
        # True positives one hot tensor
        tp = torch.zeros(len(detections))
        # False positive one hot tensor
        fp = torch.zeros(len(detections))
        total_true_boxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            # Ground truth bounding boxes which match current predicted bounding box image ID
            corresponding_gts = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            best_iou = float("-inf")
            best_gt_idx = None
            for gt_idx, gt in enumerate(corresponding_gts):
                iou = intersection_over_union(torch.tensor(detection[3:]),
                                              torch.tensor(gt[3:]),
                                              box_format=box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            # If best intersection over union for the current detection > threshold we consider this
            # detection as true positive otherwise false positive
            # If there is no detection for some gt than "for loop" for detection will not work and recall will be worse
            if best_iou > iou_threshold:
                # In case we predict already predicted bounding box we consider this as false positive
                if bboxes_amount[detection[0]][best_gt_idx] == 0:
                    bboxes_amount[detection[0]][best_gt_idx] = 1
                    tp[detection_idx] = 1
                else:
                    fp[detection_idx] = 1
            else:
                fp[detection_idx] = 1
        # Cumsum to figure out tp and fp for each prediction probability threshold
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        # Total true boxes == tp + fn
        recalls = tp_cumsum / (total_true_boxes + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        # Start point for precision-recall curve is [1, 0]
        precisions = torch.concat((torch.tensor([1]), precisions), dim=0)
        recalls = torch.concat((torch.tensor([0]), recalls), dim=0)
        average_precisions.append(torch.trapz(precisions, recalls))
    # mAP is mean of average precisions (p-r aucs)
    return sum(average_precisions) / len(average_precisions)
