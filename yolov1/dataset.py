import torch
import torchvision
from torch.utils.data import Dataset
import pathlib
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt, patches


class VOCDataset(Dataset):
    """
    PascalVOC dataset with yolo labeling for object detection task
    ...
    Attributes
    ----------
    annotations: pd.DataFrame
        Dataframe which describes yolo labeling file and corresponding image
    image_dir: pathlib.Path
        Directory path to dataset's images
    label_dir: pathlib.Path
        Directory path to yolo bounding boxes annotations
    transforms:
        Transformations to apply on image, bounding box
    s: int
        Grid cells split for the yolo model
    b: int
        Number of bounding box anchors for the yolo model
    c: int
        Number of objects classes to detect
    """

    def __init__(self,
                 descr_csv,
                 image_dir,
                 label_dir,
                 s=7, b=2, c=20,
                 transforms=None):
        self.annotations = pd.read_csv(descr_csv)
        self.image_dir = pathlib.Path(image_dir)
        self.label_dir = pathlib.Path(label_dir)
        self.transforms = transforms
        self.s = s
        self.b = b
        self.c = c

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, index):
        label_path = self.label_dir / self.annotations.iloc[index, 1]
        bboxes = []
        with open(label_path, "r") as label_file:
            for line in label_file.readlines():
                class_label, x, y, width, height = tuple(
                    map(
                        lambda num: float(num), line.strip('\n').split()
                    )
                )
                bboxes.append([class_label, x, y, width, height])

        image_path = self.image_dir / self.annotations.iloc[index, 0]
        image = Image.open(image_path)

        if self.transforms:
            # Apply augmentation to bboxes too
            image = self.transforms(image)

        # Resulting image label containing bounding boxes
        label_tensor = torch.zeros(self.s, self.s, self.c + 5 * self.b)
        for bbox in bboxes:
            class_label, x, y, width, height = bbox
            class_label = int(class_label)
            # i and j of grid cell which contains a true bounding box
            grid_cell_i, grid_cell_j = int(self.s * y), int(self.s * x)
            # y and x of center of a true bounding box
            grid_cell_y, grid_cell_x = self.s * y - grid_cell_i, self.s * x - grid_cell_j
            # width and height of a true bounding box
            grid_cell_width, grid_cell_height = self.s * width, self.s * height

            if label_tensor[grid_cell_j, grid_cell_j, 20] == 0:
                # Probability of containing object = 1
                label_tensor[grid_cell_i, grid_cell_j, 20] = 1
                # Labels of a true bounding box
                label_tensor[grid_cell_i, grid_cell_j, 21:25] = torch.tensor([
                    grid_cell_x, grid_cell_y, grid_cell_width, grid_cell_height
                ])
                # Which class the object in a true bounding box belongs to
                label_tensor[grid_cell_i, grid_cell_j, class_label] = 1
        return image, label_tensor


if __name__ == "__main__":
    # Draw some ground trues bounding boxes
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    dataset = VOCDataset(descr_csv="../archive/100examples.csv",
                         image_dir="../archive/images",
                         label_dir="../archive/labels",
                         transforms=transforms)
    image_, bboxes_ = dataset[0]
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_.permute(1, 2, 0))
    for i in range(7):
        for j in range(7):
            if bboxes_[i, j, 20] == 1:
                x, y, width, height = bboxes_[i, j, 21:25]
                rectangle_x = (x + j - width / 2) * image_.shape[2] / 7
                rectangle_y = (y + i - height / 2) * image_.shape[1] / 7
                rectangle = patches.Rectangle(
                    (rectangle_x, rectangle_y),
                    width * image_.shape[2] / 7, height * image_.shape[1] / 7,
                    linewidth=2, edgecolor='r', facecolor="none"
                )
                ax.add_patch(rectangle)
    plt.show()
