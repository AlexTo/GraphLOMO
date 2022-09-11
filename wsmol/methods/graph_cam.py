import pickle
import torch
import torch.nn as nn
from ..gnn.util import AdjacencyHelper


class GraphCAM(nn.Module):
    def __init__(self, model, embedding, target_module):
        super(GraphCAM, self).__init__()
        self.model = model
        self.model.eval()
        self.num_classes = self.model.num_classes

        with open(embedding, 'rb') as f:
            self.emb = pickle.load(f)
            self.emb = torch.from_numpy(self.emb).cuda()
        
        for module in self.model.modules():
            if module == target_module:
                module.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        self.activation = output


    def forward(self, imgs, targets=None, return_cam=False):
        f = self.model.model(imgs)

        if self.model.gt is not None:
            adj, _ = self.model.gt(self.model.A)
            adj = torch.squeeze(adj, 0)
        else:
            adj = self.model.A[0][0].detach()

        adj += torch.eye(self.num_classes).type(torch.FloatTensor).cuda()
        adj = AdjacencyHelper.transform_adj(adj)

        w = self.model.gc1(self.emb, adj)
        w = self.model.relu(w)
        w = self.model.gc2(w, adj)

        w = w.transpose(0, 1)

        if not return_cam:
            y = torch.matmul(f, w)
            return {"logits": y}
        
        _, k, _, _ = self.activation.shape
        _, nc = w.shape

        cams = torch.relu(torch.sum(self.activation.unsqueeze(1).repeat(1, nc, 1, 1, 1) * w.T.view(nc, k, 1, 1), dim=2))
        
        return cams

    def __call__(self, imgs, targets=None, return_cam=False):
        return super(GraphCAM, self).__call__(imgs, targets, return_cam)
