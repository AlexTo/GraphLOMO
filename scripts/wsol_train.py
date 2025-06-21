"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim

from wsmol.config import get_configs
from data_loaders import get_data_loader
from inference import CAMComputer
from wsmol.util import AveragePrecisionMeter, string_contains_any
import wsmol
import wsmol.methods
from tqdm import tqdm


def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.value_per_epoch = [] \
            if split == 'val' else [-np.inf if higher_is_better else np.inf]

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)


class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'classification', 'localization']
    _BEST_CRITERION_METRIC = 'classification'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "CUB_MULTI": 80,
        "ILSVRC": 1000,
        "OpenImages": 100,
        "COCO": 80,
        "CheXNet": 15,
        "VOC": 20
    }

    _CLASSIF_PROBLEM_MAPPING = {
        "CUB": "multi_class",
        "CUB_MULTI": "multi_label",
        "ILSVRC": "multi_class",
        "OpenImages": "multi_class",
        "COCO": "multi_label",
        "CheXNet": "multi_label",
        "VOC": "multi_label"
    }

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.', 'fc.'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }

    def __init__(self):
        self.args = get_configs()
        set_random_seed(self.args.seed)
        print(self.args)
        self.classif_type = self._CLASSIF_PROBLEM_MAPPING[self.args.dataset_name]
        self.performance_meters = self._set_performance_meters()
        self.reporter = self.args.reporter
        self.model = self._set_model()
        if self.classif_type == "multi_class":
            self.loss = nn.CrossEntropyLoss().cuda()
        else:
            self.loss = nn.MultiLabelSoftMarginLoss().cuda()
        self.optimizer = self._set_optimizer()
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_classes=num_classes,
            classif_type=self.classif_type,
            num_val_sample_per_class=self.args.num_val_sample_per_class)

        if self.classif_type == "multi_label":
            self.average_precision_meter = AveragePrecisionMeter()

    def _set_performance_meters(self):
        self._EVAL_METRICS += ['localization_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict

    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("Loading model {}".format(self.args.architecture))
        model = wsmol.__dict__[self.args.architecture](
            dataset_name=self.args.dataset_name,
            wsol_method=self.args.wsol_method,
            pretrained=self.args.pretrained,
            num_classes=num_classes,
            classif_type=self.classif_type,
            large_feature_map=self.args.large_feature_map,
            pretrained_path=self.args.pretrained_path,
            adl_drop_rate=self.args.adl_drop_rate,
            adl_drop_threshold=self.args.adl_threshold,
            acol_drop_threshold=self.args.acol_threshold)
        print(model)

        model = nn.DataParallel(model).cuda()
        return model

    def _set_optimizer(self):
        param_features = []
        param_classifiers = []

        def param_features_substring_list(architecture):
            for key in self._FEATURE_PARAM_LAYER_PATTERNS:
                if architecture.startswith(key):
                    return self._FEATURE_PARAM_LAYER_PATTERNS[key]
            raise KeyError("Fail to recognize the architecture {}"
                           .format(self.args.architecture))

        for name, parameter in self.model.named_parameters():
            if string_contains_any(
                    name,
                    param_features_substring_list(self.args.architecture)):
                if self.args.architecture in ('vgg16', 'inception_v3'):
                    param_features.append(parameter)
                elif self.args.architecture == 'resnet50':
                    param_classifiers.append(parameter)
            else:
                if self.args.architecture in ('vgg16', 'inception_v3'):
                    param_classifiers.append(parameter)
                elif self.args.architecture == 'resnet50':
                    param_features.append(parameter)

        optimizer = torch.optim.SGD([
            {'params': param_features, 'lr': self.args.lr},
            {'params': param_classifiers,
             'lr': self.args.lr * self.args.lr_classifier_ratio}],
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True)
        return optimizer

    def _wsol_training(self, images, target):
        if (self.args.wsol_method == 'cutmix' and
                self.args.cutmix_prob > np.random.rand(1) and
                self.args.cutmix_beta > 0):
            images, target_a, target_b, lam = wsmol.methods.cutmix(
                images, target, self.args.cutmix_beta)
            output_dict = self.model(images)
            logits = output_dict['logits']
            loss = (self.loss(logits, target_a) * lam +
                    self.loss(logits, target_b) * (1. - lam))
            return logits, loss

        if self.args.wsol_method == 'has':
            images = wsmol.methods.has(images, self.args.has_grid_size,
                                     self.args.has_drop_rate)

        output_dict = self.model(images, target)
        logits = output_dict['logits']

        if self.args.wsol_method in ('acol', 'spg'):
            loss = wsmol.methods.__dict__[self.args.wsol_method].get_loss(
                output_dict, target, spg_thresholds=self.args.spg_thresholds)
        else:
            loss = self.loss(logits, target)

        return logits, loss

    def train(self, split):
        self.model.train()
        loader = self.loaders[split]

        total_loss = 0.0
        num_images = 0
        if self.classif_type == "multi_class":
            num_correct = 0
        else:
            self.average_precision_meter.reset()

        for batch_idx, (images, target, _) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()

            if batch_idx % int(len(loader) / 10) == 0:
                print(" iteration ({} / {})".format(batch_idx + 1, len(loader)))

            logits, loss = self._wsol_training(images, target)
            if self.classif_type == "multi_class":
                pred = logits.argmax(dim=1)
                num_correct += (pred == target).sum().item()
            else:
                self.average_precision_meter.add(logits.detach(), target)

            num_images += images.size(0)

            total_loss += loss.item() * images.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_average = total_loss / float(num_images)

        self.performance_meters[split]['loss'].update(loss_average)

        train_performance = {
            "loss": loss_average
        }

        if self.args.eval_classification:
            if self.classif_type == "multi_class":
                acc = num_correct / float(num_images) * 100
                train_performance["acc"] = acc
            else:
                mAP = self.average_precision_meter.value().mean().item() * 100
                train_performance["mAP"] = mAP

            self.performance_meters[split]['classification'].update(
                acc if self.classif_type == "multi_class" else mAP)

        return train_performance

    def print_performances(self):
        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = \
                    self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    print("Split {}, metric {}, current value: {}".format(
                        split, metric, current_performance))
                    if split != 'test':
                        print("Split {}, metric {}, best value: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_value))
                        print("Split {}, metric {}, best epoch: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_epoch))

    def save_performances(self):
        log_path = os.path.join(self.args.log_folder, 'performance_log.pickle')
        with open(log_path, 'wb') as f:
            pickle.dump(self.performance_meters, f)

    def evaluate(self, epoch, split):
        print("Evaluate epoch {}, split {}".format(epoch, split))
        self.model.eval()
        loader = self.loaders[split]

        if self.classif_type == "multi_class":
            num_correct = 0
        else:
            self.average_precision_meter.reset()

        num_images = 0
        total_loss = 0.0
        for images, targets, _ in tqdm(loader, desc="Computing loss"):
            images = images.cuda()
            targets = targets.cuda()
            logits, loss = self._wsol_training(images, targets)
            total_loss += loss.item() * images.size(0)
            num_images += images.size(0)
            if self.classif_type == "multi_class":
                pred = logits.argmax(dim=1)
                num_correct += (pred == targets).sum().item()
            else:
                self.average_precision_meter.add(logits.detach(), targets)

        loss_average = total_loss / float(num_images)
        self.performance_meters[split]['loss'].update(loss_average)

        if self.args.eval_classification:
            if self.classif_type == "multi_class":
                accuracy = num_correct / float(num_images) * 100
            else:
                mAP = self.average_precision_meter.value().mean().item() * 100

            self.performance_meters[split]['classification'].update(
                accuracy if self.classif_type == "multi_class" else mAP)

        if self.args.eval_localization:
            cam_computer = CAMComputer(
                model=self.model,
                loader=self.loaders[split],
                metadata_root=os.path.join(self.args.metadata_root, split),
                mask_root=self.args.mask_root,
                iou_threshold_list=self.args.iou_threshold_list,
                dataset_name=self.args.dataset_name,
                split=split,
                cam_curve_interval=self.args.cam_curve_interval,
                multi_contour_eval=self.args.multi_contour_eval,
                log_folder=self.args.log_folder,
                classif_type=self.classif_type,
                crop_size=self.args.crop_size
            )

            cam_performance = cam_computer.compute_and_evaluate_cams()

            if self.args.multi_iou_eval or self.args.dataset_name == 'OpenImages':
                loc_score = np.average(cam_performance)
            else:
                loc_score = cam_performance[self.args.iou_threshold_list.index(
                    50)]

            self.performance_meters[split]['localization'].update(loc_score)

            if self.args.dataset_name in ('CUB', 'ILSVRC'):
                for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                    self.performance_meters[split][
                        'localization_IOU_{}'.format(IOU_THRESHOLD)].update(
                        cam_performance[idx])

    def _torch_save_model(self, filename, epoch):
        torch.save({'architecture': self.args.architecture,
                    'epoch': epoch,
                    'best_criterion_metric': self._BEST_CRITERION_METRIC,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.args.log_folder, filename))

    def save_checkpoint(self, epoch, split):
        if (self.performance_meters[split][self._BEST_CRITERION_METRIC]
                .best_epoch) == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('best'), epoch)
        if self.args.epochs == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('last'), epoch)

        if self.args.checkpoint_interval > 0 and epoch % self.args.checkpoint_interval == 0:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format(f'epoch_{epoch}'), epoch)

    def report_train(self, train_performance, epoch, split='train'):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        if "acc" in train_performance:
            reporter_instance.add(
                key='{split}/acc'.format(split=split),
                val=train_performance['acc'])
        if "mAP" in train_performance:
            reporter_instance.add(
                key='{split}/mAP'.format(split=split),
                val=train_performance['mAP'])

        reporter_instance.add(
            key='{split}/loss'.format(split=split),
            val=train_performance['loss'])
        reporter_instance.write()

    def report(self, epoch, split):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        for metric in self._EVAL_METRICS:
            reporter_instance.add(
                key='{split}/{metric}'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].current_value)
            reporter_instance.add(
                key='{split}/{metric}_best'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].best_value)
        reporter_instance.write()

    def adjust_learning_rate(self, epoch):
        if epoch != 0 and epoch % self.args.lr_decay_frequency == 0 and epoch < 60:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1

    def load_checkpoint(self, checkpoint_type):
        if checkpoint_type not in ('best', 'last'):
            raise ValueError("checkpoint_type must be either best or last.")
        checkpoint_path = os.path.join(
            self.args.log_folder,
            self._CHECKPOINT_NAME_TEMPLATE.format(checkpoint_type))
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))


def main():
    trainer = Trainer()

    print("===========================================================")
    print("Start epoch 0 ...")
    trainer.evaluate(epoch=0, split='val')
    trainer.print_performances()
    trainer.report(epoch=0, split='val')
    trainer.save_checkpoint(epoch=0, split='val')
    print("Epoch 0 done.")

    for epoch in range(1, trainer.args.epochs + 1):
        print("===========================================================")
        print("Start epoch {} ...".format(epoch))
        trainer.adjust_learning_rate(epoch)
        train_performance = trainer.train(split='train')
        trainer.report_train(train_performance, epoch, split='train')

        trainer.evaluate(epoch, split='val')
        trainer.print_performances()
        trainer.report(epoch, split='val')

        trainer.save_checkpoint(epoch, split='train')
        print("Epoch {} done.".format(epoch))

    print("===========================================================")
    print("Final epoch evaluation on test set ...")

    trainer.load_checkpoint(checkpoint_type=trainer.args.eval_checkpoint_type)
    trainer.evaluate(trainer.args.epochs, split='test')
    trainer.print_performances()
    trainer.report(trainer.args.epochs, split='test')
    trainer.save_performances()


if __name__ == '__main__':
    main()
