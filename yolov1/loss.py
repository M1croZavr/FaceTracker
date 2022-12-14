import torch
from torch import nn
from metrics import intersection_over_union


class YoloLoss(nn.Module):
    """
    Yolo version 1 loss neural network module
    ...
    Attributes
    ----------
    s: int
        Grid cells split for the yolo model
    b: int
        Number of bounding box anchors for the yolo model
    c: int
        Number of objects classes to detect
    lambda_coord: float
        Regularization coefficient for bounding box predictions
    lambda_no_obj: float
        Regularization coefficient for no object in bounding box prediction
    mse: nn.Module
        Mean squared error module with 'sum' reduction
    """

    def __init__(self, s=7, b=2, c=20):
        super(YoloLoss, self).__init__()
        self.s = s
        self.b = b
        self.c = c
        self.lambda_coord = 5
        self.lambda_no_obj = 0.5
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, targets):
        predictions = predictions.view(-1, self.s, self.s, self.c + self.b * 5)
        iou_anchor1 = intersection_over_union(
            predictions[..., self.c + 1:self.c + 5], targets[..., self.c + 1:self.c + 5], box_format="midpoint"
        )
        iou_anchor2 = intersection_over_union(
            predictions[..., self.c + 6:self.c + 10], targets[..., self.c + 1:self.c + 5], box_format="midpoint"
        )
        # iou_anchor shape: bs x 7 x 7 x 1
        # Concatenate along the new outer dim iou result of 2 anchors
        anchors_iou = torch.concat(
            (iou_anchor1.unsqueeze(dim=0), iou_anchor2.unsqueeze(dim=0)),
            dim=0
        )
        # Maximum between iou1 and iou2 elements (the outer dim) and corresponding indexes (0 or 1 anchor) in case of 2
        iou_maxes, best_box = torch.max(anchors_iou, dim=0)
        # Is there a bbox in a grid cell from true data (0 or 1)
        exists_box = targets[..., self.c:self.c + 1]

        # Bbox coordinates (x, y, w, h) loss component
        bbox_predictions = exists_box * (
                (best_box * predictions[..., self.c + 6:self.c + 10]) +
                ((1 - best_box) * predictions[..., self.c + 1:self.c + 5])
        )
        bbox_targets = exists_box * targets[..., self.c + 1:self.c + 5]
        # Extract the square root of the width and height
        bbox_predictions[..., 2:4] = torch.sqrt(torch.abs(bbox_predictions.clone()[..., 2:4]) + 1e-5) \
                                     * torch.sign(bbox_predictions.clone()[..., 2:4])
        bbox_targets[..., 2:4] = torch.sqrt(torch.abs(bbox_targets[..., 2:4]) + 1e-5) \
                                 * torch.sign(bbox_targets[..., 2:4])
        # Flattening to bs * 7 * 7 x 4, aggregating each bounding box for mse loss
        bbox_loss = self.mse(
            torch.flatten(bbox_predictions, start_dim=0, end_dim=-2),
            torch.flatten(bbox_targets, start_dim=0, end_dim=-2)
        )

        # Object in bbox (C) loss component
        object_predictions = ((best_box * predictions[..., self.c + 5: self.c + 6])
                              + ((1 - best_box) * predictions[..., self.c:self.c + 1]))
        object_loss = self.mse(
            torch.flatten(exists_box * object_predictions),
            torch.flatten(exists_box)
        )

        # No object bbox (C) loss component
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.c:self.c + 1], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., self.c:self.c + 1], start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.c + 5:self.c + 6], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., self.c:self.c + 1], start_dim=1)
        )

        # Classes loss component
        classes_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.c], end_dim=-2),
            torch.flatten(exists_box * targets[..., :self.c], end_dim=-2)
        )

        # Total loss
        loss = self.lambda_coord * bbox_loss + object_loss + self.lambda_no_obj * no_object_loss + classes_loss
        return loss


if __name__ == "__main__":
    yolo_loss = YoloLoss()
    predictions_tens = torch.randn(10, 7, 7, 30, requires_grad=True)
    true_tens = torch.randn(10, 7, 7, 25)
    with torch.autograd.set_detect_anomaly(True):
        loss = yolo_loss(predictions_tens, true_tens)
        loss.backward()
        print(loss.item())
