import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
import random
import torchvision.models as models


def load_resnet(model_name, pretrained=True):
    if model_name == 'resnet18':
        return models.resnet18(pretrained=pretrained)
    if model_name == 'resnet34':
        return models.resnet34(pretrained=pretrained)
    if model_name == 'resnet50':
        return models.resnet50(pretrained=pretrained)
    if model_name == 'resnet101':
        return models.resnet101(pretrained=pretrained)
    if model_name == 'resnet152':
        return models.resnet152(pretrained=pretrained)
    if model_name == 'resnext50_32x4d':
        return models.resnext50_32x4d(pretrained=pretrained)
    if model_name == 'resnext101_32x8d':
        return models.resnext101_32x8d(pretrained=pretrained)
    if model_name == 'wide_resnet50_2':
        return models.wide_resnet50_2(pretrained=pretrained)
    if model_name == 'wide_resnet101_2':
        return models.wide_resnet101_2(pretrained=pretrained)
    if model_name == 'resnet18_swsl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_swsl')
    if model_name == 'resnet50_swsl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
    if model_name == 'resnext50_32x4d_swsl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
    if model_name == 'resnext101_32x4d_swsl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_swsl')
    if model_name == 'resnext101_32x8d_swsl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_swsl')
    if model_name == 'resnext101_32x16d_swsl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')
    if model_name == 'resnet18_ssl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_ssl')
    if model_name == 'resnet50_ssl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_ssl')
    if model_name == 'resnext50_32x4d_ssl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
    if model_name == 'resnext101_32x4d_ssl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_ssl')
    if model_name == 'resnext101_32x8d_ssl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_ssl')
    if model_name == 'resnext101_32x16d_ssl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_ssl')
    return None


def load_vgg(model_name, pretrained=True):
    if model_name == 'vgg11':
        return models.vgg11(pretrained=pretrained)
    if model_name == 'vgg11_bn':
        return models.vgg11_bn(pretrained=pretrained)
    if model_name == 'vgg13':
        return models.vgg13(pretrained=pretrained)
    if model_name == 'vgg13_bn':
        return models.vgg13_bn(pretrained=pretrained)
    if model_name == 'vgg16':
        return models.vgg16(pretrained=pretrained)
    if model_name == 'vgg16_bn':
        return models.vgg16_bn(pretrained=pretrained)
    if model_name == 'vgg19':
        return models.vgg19(pretrained=pretrained)
    if model_name == 'vgg19_bn':
        return models.vgg19_bn(pretrained=pretrained)
    return None


def topk(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
    return topk_data_sort, topk_index_sort


class Cutout(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)


class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [
            input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop(
            (offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize(
            (self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(
            x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(
            x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    def __str__(self):
        return self.__class__.__name__


class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = np.empty([0], dtype=float)
        self.targets = np.empty([0], dtype=int)

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if torch.is_tensor(output):
            output = output.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        if output.ndim == 1:
            output = output.reshape(-1, 1)
        else:
            assert output.ndim == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        else:
            assert target.ndim == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.size > 0:
            assert target.shape[1] == self.targets.shape[1], \
                'dimensions for output should match previously added examples.'

        self.scores = np.vstack([self.scores, output.astype(
            np.float)]) if self.scores.size > 0 else output.astype(np.float)
        self.targets = np.vstack([self.targets, target.astype(
            np.int)]) if self.targets.size > 0 else target.astype(np.int)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """
        if self.scores.size == 0:
            return 0

        ap = np.zeros(self.scores.shape[1])
        # compute average precision for each class
        for k in range(self.scores.shape[1]):
            # compute average precision
            ap[k] = self.average_precision(
                self.scores[:, k], self.targets[:, k], self.difficult_examples)
        return ap

    def average_precision(self, output, target, difficult_examples=False):

        # sort examples
        indices = output.argsort()[::-1]

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.size == 0:
            return 0
        scores = self.scores
        targets = self.targets
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets
        targets[targets == -1] = 0
        n, c = self.scores.shape
        scores = np.zeros((n, c)) - 1
        index = topk(self.scores, k)[1]
        tmp = self.scores
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1


class AdjacencyHelper:

    @staticmethod
    def load_adj(num_classes, t=0.4, adj_files=None, add_identity=False):
        _adj_stack = torch.eye(num_classes).type(
            torch.FloatTensor).unsqueeze(-1) if add_identity else torch.Tensor([])

        for adj_file in adj_files:
            if '_emb' in adj_file:
                _adj = AdjacencyHelper.gen_emb_A(adj_file)
            else:
                _adj = AdjacencyHelper.gen_A(num_classes, adj_file, t)

            _adj = torch.from_numpy(_adj).type(torch.FloatTensor)
            _adj_stack = torch.cat([_adj_stack, _adj.unsqueeze(-1)], dim=-1)

        return _adj_stack.permute(2, 0, 1)

    @staticmethod
    def gen_A(num_classes, adj_file, t=None):
        import pickle
        result = pickle.load(open(adj_file, 'rb'))
        _adj = result['adj']
        _nums = result['nums']
        _nums = _nums[:, np.newaxis]
        _adj = _adj / (_nums + 1e-6)

        if t is not None and t > 0.0:
            _adj[_adj < t] = 0
            _adj[_adj >= t] = 1

        _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
        return _adj

    @staticmethod
    def gen_emb_A(adj_file, t=None):
        import pickle
        result = pickle.load(open(adj_file, 'rb'))
        _adj = result['adj']
        mean_v = _adj.mean()
        std_v = _adj.std()
        t = mean_v - std_v if t is None else t

        if t is not None and t > 0.0:
            _adj[_adj < t] = 0
            _adj[_adj >= t] = 1

        _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
        return _adj

    @staticmethod
    def transform_adj(A, is_stack=False):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    @staticmethod
    def batched_target_to_adj(target):
        batch_size = target.size()[0]
        n = target.size()[1]
        starget = torch.reshape(target, [batch_size * n])
        zero = torch.zeros(batch_size * n)
        ztarget = torch.where(starget < 0, starget, zero)
        indices = torch.nonzero(ztarget, as_tuple=True)[0]
        smask_adj = torch.ones((batch_size * n, n))
        smask_adj[indices] = 0
        mask_adj = torch.reshape(smask_adj, [batch_size, n, n])
        mask_adj_r = torch.rot90(mask_adj, 1, [1, 2])
        adj = (mask_adj.bool() & mask_adj_r.bool() & torch.logical_not(
            torch.diag_embed(torch.ones((batch_size, n))).bool())).float()
        return adj

    @staticmethod
    def batched_target_to_nums(target):
        batch_size = target.size()[0]
        n = target.size()[1]
        zero = torch.zeros(batch_size, n)
        ztarget = torch.where(target > 0, target, zero)
        return torch.sum(ztarget, 0)

    @staticmethod
    def normalise_adj(A, t=0.4):
        # Thresholding
        A.masked_fill_(A < t, 0.0)
        A.masked_fill_(A >= t, 1.0)
        # Normalisation
        A = torch.div(torch.mul(A, 0.25), torch.add(
            A.sum(0, keepdim=True), 1e-6))
        # Add identity matrix
        mask = torch.eye(A.shape[0]).bool()
        A.masked_fill_(mask, 1.0)

        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    @staticmethod
    def batched_adj_to_freq(adj):
        return torch.sum(adj, 0)
