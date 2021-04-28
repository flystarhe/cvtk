# https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/segmentation.py
from torchvision.models.segmentation.segmentation import *

__all__ = ["fcn_resnet50", "fcn_resnet101",
           "deeplabv3_resnet50", "deeplabv3_resnet101",
           "deeplabv3_mobilenet_v3_large", "lraspp_mobilenet_v3_large"]
