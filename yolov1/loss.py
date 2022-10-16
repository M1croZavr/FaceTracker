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
    lambda_noobj: float
        Regularization coefficient for no object in bounding box prediction
    mse: nn.Moduel
        Mean squared error module with 'sum' reduction
    """
    def __init__(self, s=7, b=2, c=20):
        super(YoloLoss, self).__init__()
        self.s = s
        self.b = b
        self.c = c
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, targets):
        predictions = predictions.view(-1, self.s, self.s, self.c + self.b * 5)
        iou_anchor1 = intersection_over_union(
            predictions[..., 21:25], targets[..., 21:25], box_format="midpoint"
        )
        iou_anchor2 = intersection_over_union(
            predictions[..., 26:30], targets[..., 21:25], box_format="midpoint"
        )
        # iou_anchor shape: bs x 7 x 7 x 1
        # Concatenate along the new outer dim iou result of 2 anchors
        anchors_iou = torch.concat((iou_anchor1.unsqueeze(dim=0), iou_anchor2.unsqueeze(dim=0)), dim=0)
        # Maximum between iou1 and iou2 elements (the outer dim) and corresponding indexes (0 or 1 anchor) in case of 2
        iou_maxes, best_box = torch.max(anchors_iou, dim=0)
        # Is there a bbox in a grid cell from true data (0 or 1)
        exists_box = targets[..., 20:21]

        # Bbox coordinates (x, y, w, h) loss component
        bbox_predictions = exists_box * (
                (best_box * predictions[..., 26:30]) + ((1 - best_box) * predictions[..., 21:25]))
        bbox_targets = exists_box * targets[..., 21:25]
        # Extract the square root of the width and height
        bbox_predictions[..., 2:4] = torch.sqrt(torch.abs(bbox_predictions[..., 2:4]) + 1e-5) \
            * torch.sign(bbox_predictions[..., 2:4])
        bbox_targets[..., 2:4] = torch.sqrt(torch.abs(bbox_targets[..., 2:4]) + 1e-5) \
            * torch.sign(bbox_targets[..., 2:4])
        # Flattening to bs * 7 * 7 x 4
        bbox_loss = self.mse(
            torch.flatten(bbox_predictions, start_dim=0, end_dim=-2),
            torch.flatten(bbox_targets, start_dim=0, end_dim=-2)
        )

        # Object in bbox (C) loss component
        object_predictions = (best_box * predictions[..., 25:26]) + ((1 - best_box) * predictions[..., 20:21])
        object_loss = self.mse(
            torch.flatten(exists_box * object_predictions),
            torch.flatten(exists_box)
        )

        # No object bbox (C) loss component
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., 20:21], start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., 20:21], start_dim=1)
        )

        # Classes loss component
        classes_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * targets[..., :20], end_dim=-2)
        )

        # Total loss
        loss = self.lambda_coord * bbox_loss + object_loss + self.lambda_noobj * no_object_loss + classes_loss
        return loss


if __name__ == "__main__":
    yolo_loss = YoloLoss()
    predictions_tens = torch.randn(10, 7, 7, 30)
    true_tens = torch.randn(10, 7, 7, 25)
    print(yolo_loss(predictions_tens, true_tens))
