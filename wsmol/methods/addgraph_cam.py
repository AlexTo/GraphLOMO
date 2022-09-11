import torch
import torch.nn as nn


class ADDGraphCAM(nn.Module):
    def __init__(self, model):
        super(ADDGraphCAM, self).__init__()
        self.model = model
        self.model.eval()

    def forward(self, imgs, targets=None, return_cam=False):

        if not return_cam:
            logits = self.model(imgs)
            return {"logits": logits}

        model = self.model

        x = model.forward_feature(imgs)
        x_prime = model.conv_transform(x)  # 1024 * 14 * 14

        classif_out = model.fc(x)  # 80 * 14 * 14

        v = model.forward_sam(x)
        z = model.forward_dgcn(v)
        z = v + z  # 1024 * 80

        b, k, nc = z.shape

        cams = torch.relu(torch.mean(x_prime.unsqueeze(1).repeat(
            1, nc, 1, 1, 1) * z.transpose(1, 2).view(b, nc, k, 1, 1), dim=2))  # + classif_out)
        return cams

    def __call__(self, imgs, targets=None, return_cam=False):
        return super(ADDGraphCAM, self).__call__(imgs, targets, return_cam)
