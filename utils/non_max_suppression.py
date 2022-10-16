import torch
from typing import List
from ..metrics import intersection_over_union


def non_max_suppression(boxes_models: List[torch.Tensor],
                        threshold_iou: float,
                        threshold_prob: float,
                        box_format: str = "corners"):
    """
    Non-max suppression proposals filtering technique
    Parameters:
        boxes_models (List[torch.Tensor]): List of bounding boxes proposed by a model
        threshold_iou (float): Intersection over union threshold
        threshold_prob (float): Probability threshold
        box_format (str): On which format bounding boxes are passed.
         Corner points or middle point with height and width. (x1, y1, x2, y2) in case of corners and
         (x, y, width, height) in case of midpoint.
    Returns:
        bboxes_after_nms (List): List of bounding boxes after non-max suppression
    """
    # Filtering by probability
    bboxes = [bbox for bbox in boxes_models if bbox[1] > threshold_prob]
    # Sorting by probability, highest is first (descending)
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    while bboxes:
        # Pick the bounding box with the highest probability
        chosen_bbox = bboxes.pop(0)
        # Remain bbox if its of a different class or iou between chosen_bbox < threshold
        bboxes = [bbox for bbox in bboxes if (bbox[0] != chosen_bbox[0]) or
                  (intersection_over_union(chosen_bbox[2:], bbox[2:], box_format)) < threshold_iou]
        bboxes_after_nms.append(chosen_bbox)
    return bboxes_after_nms
