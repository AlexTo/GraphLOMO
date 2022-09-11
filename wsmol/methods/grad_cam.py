import torch
import torch.nn as nn


class GradCAM(nn.Module):
    def __init__(self, model, target_module):
        super(GradCAM, self).__init__()
        self.model = model
        self.model.eval()

        for module in self.model.modules():
            if module == target_module:
                module.register_forward_hook(self.forward_hook)
                module.register_full_backward_hook(self.backward_hook)

        self.gradient = None

    def backward_hook(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def forward_hook(self, module, input, output):
        self.activation = output

    def forward(self, imgs, targets=None, return_cam=False):

        logits = self.model(imgs)
        if not return_cam:
            return {"logits": logits}

        cams = []

        A = self.activation
        b, k, w, h = A.size()
        num_classes = targets.shape[1]
        for c in range(num_classes):
            logits_c = logits[:, c]
            targets_c = targets[:, c]
            if targets_c.sum() == 0:
                cam = torch.zeros(b, w, h).cuda()
            else:
                yc = torch.sum(logits_c * targets_c)
                self.model.zero_grad()
                yc.backward(retain_graph=True)
                gradient = self.gradient
                wc = torch.mean(gradient, axis=(2, 3))
                cam = torch.relu(
                    torch.sum(wc.view(b, k, 1, 1) * A, dim=1))
            cams.append(cam)
        return torch.stack(cams, dim=1)

    def __call__(self, imgs, targets=None, return_cam=False):
        return super(GradCAM, self).__call__(imgs, targets, return_cam)
