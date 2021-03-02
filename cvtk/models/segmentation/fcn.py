from collections import OrderedDict
from typing import Dict

from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet


class IntermediateLayerGetter(nn.ModuleDict):

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FCNHead(nn.Sequential):

    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


class FCN(nn.Module):

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            result["aux"] = x

        return result


def segm_model(backbone_name, pretrained, num_classes, aux_loss):
    if "resnet" in backbone_name:
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained,
            replace_stride_with_dilation=[False, True, True])
        out_layer = "layer4"
        out_inplanes = 2048
        aux_layer = "layer3"
        aux_inplanes = 1024
    else:
        raise NotImplementedError("backbone {} is not supported as of now".format(backbone_name))

    return_layers = {out_layer: "out"}
    if aux_loss:
        return_layers[aux_layer] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    classifier = FCNHead(out_inplanes, num_classes)

    aux_classifier = None
    if aux_loss:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)
    return model
