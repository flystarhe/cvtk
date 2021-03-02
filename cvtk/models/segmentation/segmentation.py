# https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/segmentation.py
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import fcn_resnet101
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import deeplabv3_resnet101

__all__ = ["fcn_resnet50", "fcn_resnet101", "deeplabv3_resnet50", "deeplabv3_resnet101"]
