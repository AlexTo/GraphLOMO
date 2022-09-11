import torch
import torch.nn as nn
from .util import AdjacencyHelper, load_resnet, load_vgg
from .gnn import GConv, GTLayer


class Net(nn.Module):
    def __init__(self, model, num_classes, model_features=None):
        super(Net, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.fc = None if model_features is None or model_features == num_classes else nn.Linear(
            model_features, num_classes)

    def forward(self, img, ext=None):
        """Performs neural network operations.

        Args:
            img: input image (B, 3, W, H)
            ext: extra input for compatibility

        Returns:
            output (B, num_classes)

        """
        x = self.model(img)  # (B, 2048)
        return x if self.fc is None else self.fc(x)

    def get_config_optim(self, lr, lrg, add_weight_decay=False, weight_decay=1e-4, skip_list=()):
        if add_weight_decay:
            decay = []
            no_decay = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'lr': lr, 'weight_decay': 0.},
                {'params': decay, 'lr': lr, 'weight_decay': weight_decay}
            ]

        return [
            {'params': self.model.parameters(), 'lr': lr}
        ]


class GraphNet(nn.Module):
    def __init__(self, model, num_classes, emb_features=300, gc1_features=1024, gc2_features=2048, t=0.4, adj_files=None, gtn=False):
        super(GraphNet, self).__init__()
        self.model = model
        self.num_classes = num_classes

        self.gc1 = GConv(emb_features, gc1_features)
        self.gc2 = GConv(gc1_features, gc2_features)
        self.relu = nn.LeakyReLU(0.2)

        self.A = nn.Parameter(AdjacencyHelper.load_adj(
            num_classes, t=t, adj_files=adj_files, add_identity=gtn).unsqueeze(0))
        self.gt = GTLayer(self.A.shape[1], 1, first=True) if gtn else None

    def forward(self, img, emb):
        """Performs neural network operations.

        Args:
            img: input image (B, 3, W, H)
            emb: input embedding (B, num_classes, emb_length)

        Returns:
            output (B, num_classes)

        """
        f = self.forward_features(img)  # (B, C_out=2048)
        y = self.forward_gcn(f, emb)  # (B, num_classes)
        return y

    def forward_features(self, img):
        """Performs feature extraction.

        Args:
            img: input image (B, 3, W, H)

        Returns:
            output (B, C_out=2048)

        """
        return self.model(img)  # (B, 2048)

    def forward_gcn(self, x, emb):
        """Performs graph convolutional network operations.

        Args:
            x: input feature (B, C_out=2048)
            emb: input embedding (B, num_classes, emb_length)

        Returns:
            output (B, num_classes)

        """
        if self.gt is not None:
            adj, _ = self.gt(self.A)
            adj = torch.squeeze(adj, 0)
        else:
            adj = self.A[0][0].detach()

        adj += torch.eye(self.num_classes).type(torch.FloatTensor).cuda()
        adj = AdjacencyHelper.transform_adj(adj)  # (num_classes, num_classes)

        w = self.gc1(emb[0], adj)  # (num_classes, gc1_features=1024)
        w = self.relu(w)  # (num_classes, gc1_features=1024)
        w = self.gc2(w, adj)  # (num_classes, gc2_features = 2048)

        w = w.transpose(0, 1)  # (gc2_features, num_classes)
        y = torch.matmul(x, w)  # (B, num_classes)
        return y  # (B, num_classes)

    def get_config_optim(self, lr, lrg, add_weight_decay=False, weight_decay=1e-4, skip_list=()):
        if add_weight_decay:
            decay = []
            no_decay = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'lr': lr, 'weight_decay': 0.},
                {'params': decay, 'lr': lr, 'weight_decay': weight_decay},
                {'params': self.gc1.parameters(), 'lr': lrg},
                {'params': self.gc2.parameters(), 'lr': lrg},
            ]

        return [
            {'params': self.model.parameters(), 'lr': lr},
            {'params': self.gc1.parameters(), 'lr': lrg},
            {'params': self.gc2.parameters(), 'lr': lrg},
        ]


def resnet(num_classes, arch='resnext50_32x4d_swsl', pretrained=True, model_features=2048):
    model = load_resnet(arch, pretrained=pretrained)
    model = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        nn.MaxPool2d(14, 14),
        nn.Flatten(1)
    )
    return Net(model, num_classes, model_features=2048)


def vgg(num_classes, arch='vgg16', pretrained=True):
    model = load_vgg(arch, pretrained=pretrained)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return Net(model, num_classes)


def efficientnet(num_classes, arch='efficientnet-b6', pretrained=True):
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained(
        arch, num_classes=num_classes) if pretrained else EfficientNet.from_name(arch, num_classes=num_classes)
    return Net(model, num_classes)


def build_net(num_classes, arch='resnext50_32x4d_swsl', pretrained=True, graph=False, gtn=False, t=0.4, adj_files=None, emb_features=300, model_features=2048):
    if 'vgg' in arch:
        net = vgg(model_features if graph else num_classes,
                  arch=arch, pretrained=pretrained)
    elif 'efficientnet' in arch:
        net = efficientnet(
            model_features if graph else num_classes, arch=arch, pretrained=pretrained)
    else:
        net = resnet(num_classes, arch=arch, pretrained=pretrained)

    if graph:
        return GraphNet(net.model, num_classes, emb_features=emb_features, gc1_features=model_features//2, gc2_features=model_features, t=t, adj_files=adj_files, gtn=gtn)
    else:
        return net
