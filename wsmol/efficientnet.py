import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Union

from .methods.grad_cam import GradCAM
from .gnn import models


def load_pretrained_model(model, path=None):
    strict = True
    if path is not None:
        print(f"Loaded model from {path}")
        state_dict = torch.load(path)['state_dict']
        model.load_state_dict(state_dict, strict=strict)
    return model


class EfficientNetGradCam(nn.Module):
    def __init__(self, classif_type="multi_class", num_classes=80, **kwargs):
        super(EfficientNetGradCam, self).__init__()
        self.num_classes = num_classes
        self.classif_type = classif_type
        self.model = models.efficientnet(
            num_classes=num_classes, arch="efficientnet-b4")
        self.grad_cam = GradCAM(
            self.model, target_module=self.model.model._conv_head)

    def forward(self, imgs, labels=None, return_cam=False):
        return self.grad_cam(imgs, labels, return_cam)

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool):
        return self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination, prefix, keep_vars)


def efficientnet(wsol_method, classif_type, num_classes, pretrained=False, pretrained_path=None, **kwargs):
    model = {'grad_cam': EfficientNetGradCam}[
        wsol_method](classif_type, num_classes, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, path=pretrained_path)
    return model
