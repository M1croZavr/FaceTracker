import torch


def intersection_over_union(boxes_model: torch.Tensor,
                            boxes_true: torch.Tensor,
                            box_format: str = "midpoint"):
    """
    Intersection over union object detection metric
    Parameters:
        boxes_model (torch.Tensor): Box coordinates of model predictions. Shape of batch_size x 4.
        boxes_true (torch.Tensor): Box coordinates of true data. Shape of batch_size x 4.
        box_format (str): On which format bounding boxes are passed.
         Corner points or middle point with height and width. (x1, y1, x2, y2) in case of corners and
         (x, y, width, height) in case of midpoint.
    Returns:
        iou (torch.Tensor): Intersection over union metric result for model boxes predictions for each example
        in batch.
    """
    assert(box_format in {"corners", "midpoint"}), "Format must be corners or midpoint"
    if box_format == "corners":
        box1_x1 = boxes_model[..., 0:1]
        box1_y1 = boxes_model[..., 1:2]
        box1_x2 = boxes_model[..., 2:3]
        box1_y2 = boxes_model[..., 3:4]

        box2_x1 = boxes_true[..., 0:1]
        box2_y1 = boxes_true[..., 1:2]
        box2_x2 = boxes_true[..., 2:3]
        box2_y2 = boxes_true[..., 3:4]
    elif box_format == "midpoint":
        width = boxes_model[..., 2:3]
        height = boxes_model[..., 3:4]
        box1_x1 = boxes_model[..., 0:1] - width / 2
        box1_x2 = boxes_model[..., 0:1] + width / 2
        box1_y1 = boxes_model[..., 1:2] - height / 2
        box1_y2 = boxes_model[..., 1:2] + height / 2

        width = boxes_true[..., 2:3]
        height = boxes_true[..., 3:4]
        box2_x1 = boxes_true[..., 0:1] - width / 2
        box2_x2 = boxes_true[..., 0:1] + width / 2
        box2_y1 = boxes_true[..., 1:2] - height / 2
        box2_y2 = boxes_true[..., 1:2] + height / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    # In case there is no intersection at least one of multipliers will be negative
    intersection = torch.clamp((x2 - x1), min=0) * torch.clamp((y2 - y1), min=0)

    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    iou = intersection / (box1_area + box2_area - intersection + 1e-4)
    return iou


if __name__ == "__main__":
    tens1 = torch.randn(10, 7, 7, 4)
    tens2 = torch.randn(10, 7, 7, 4)
    iou_t1_t2 = intersection_over_union(tens1, tens2, "corners")
    print(iou_t1_t2.shape)
    print(iou_t1_t2)
