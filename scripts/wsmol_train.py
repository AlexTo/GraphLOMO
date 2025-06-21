import argparse
import os
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from wsmol.datasets import COCO2014
from wsmol.datasets import Voc2007Classification
from wsmol.gnn.logger import init_log
from wsmol.gnn.engine import GraphMultiLabelEngine
from wsmol.gnn.models import build_net


dataset_mappings = {
    "coco": {"dataset_class": COCO2014, "num_classes": 80},
    "voc": {"dataset_class": Voc2007Classification, "num_classes": 20},
}

parser = argparse.ArgumentParser(
    description="Graph Multi-Label Classification Training"
)
parser.add_argument("--data-dir", help="path to dataset (e.g. dataset/COCO")
parser.add_argument(
    "--metadata-dir", help="path to dataset metadata (e.g. metadata/COCO"
)
parser.add_argument("--dataset", help="name of the dataset e.g coco, voc")
parser.add_argument(
    "--image-size",
    "-i",
    default=448,
    type=int,
    metavar="N",
    help="image size (default: 448)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 8)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--epoch_step",
    default=[],
    type=int,
    nargs="+",
    help="number of epochs to change learning rate",
)
parser.add_argument(
    "--device_ids", default=[0], type=int, nargs="+", help="gpu devices to use"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=16,
    type=int,
    metavar="N",
    help="mini-batch size (default: 16)",
)
parser.add_argument(
    "-bt",
    "--batch-size-test",
    default=None,
    type=int,
    metavar="N",
    help="mini-batch size for test (default: 16)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.0045,
    type=float,
    metavar="LR",
    help="learning rate for visual representation learning",
)
parser.add_argument(
    "--lrg",
    "--learning-rate-graph",
    default=0.045,
    type=float,
    metavar="LR",
    help="learning rate for graph representation learning",
)
parser.add_argument(
    "--lrd",
    "--learning-rate-decay",
    default=0.1,
    type=float,
    metavar="LRD",
    help="learning rate decay",
)
parser.add_argument(
    "--lrb",
    "--learning-rate-best",
    dest="lrb",
    action="store_true",
    help="use the previous best when lr is changed",
)
parser.add_argument(
    "--lrs",
    "--learning-rate-scheduler",
    dest="lrs",
    action="store_true",
    help="use lr scheduler for setting learing rates",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=0,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--embedding", type=str, metavar="EMB", help="path to embedding (default: glove)"
)
parser.add_argument(
    "--embedding-length",
    default=300,
    type=int,
    metavar="EMB",
    help="embedding length (default: 300)",
)
parser.add_argument(
    "-o",
    "--optim",
    dest="optim",
    default="SGD",
    type=str,
    metavar="O",
    help="Optimizer (default: SGD)",
)
parser.add_argument(
    "-a",
    "--arch",
    dest="arch",
    default="resnet50",
    type=str,
    metavar="A",
    help="backbone architecture to be used (default: resnet50)",
)
parser.add_argument(
    "-aug", "--augmentation", dest="aug", action="store_true", help="use rand + cutout"
)
parser.add_argument(
    "-g", "--graph", dest="graph", action="store_true", help="use graph convolutions"
)
parser.add_argument(
    "-gtn",
    "--transformer",
    dest="gtn",
    action="store_true",
    help="use attention-driven dynamic graph",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "-t",
    "--adj-threshold",
    default=0.4,
    type=float,
    metavar="T",
    help="Adj threshod (default: 0.4",
)
parser.add_argument(
    "-adj",
    "--adj-files",
    default=["metadata/COCO/topology/coco_adj.pkl"],
    type=str,
    nargs="+",
    help="Adj files (default: [coco_adj.pkl])",
)
parser.add_argument(
    "-n",
    "--exp-name",
    dest="exp_name",
    default="coco",
    type=str,
    metavar="COCO",
    help="Name of experiment to have different location to save checkpoints",
)
parser.add_argument(
    "-nt", "--neptune", dest="neptune", action="store_true", help="run with neptune"
)
parser.add_argument(
    "--neptune_path", default=".neptune", help="path to neptune config file"
)


def main():
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    init_log(args)

    dataset_class = dataset_mappings[args.dataset]["dataset_class"]
    train_dataset = dataset_class(
        args.data_dir, args.metadata_dir, phase="train", emb_name=args.embedding
    )
    val_dataset = dataset_class(
        args.data_dir, args.metadata_dir, phase="val", emb_name=args.embedding
    )
    num_classes = dataset_mappings[args.dataset]["num_classes"]

    model = build_net(
        num_classes=num_classes,
        arch=args.arch,
        graph=args.graph,
        gtn=args.gtn,
        t=args.adj_threshold,
        adj_files=args.adj_files,
        emb_features=args.embedding_length,
    )

    criterion = nn.MultiLabelSoftMarginLoss()

    optimizer = torch.optim.SGD(
        model.get_config_optim(args.lr, args.lrg),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler()

    model_path = f"checkpoint/{args.dataset}/{args.exp_name}"

    os.makedirs(model_path, exist_ok=True)

    state = {
        "batch_size": args.batch_size,
        "batch_size_test": (
            args.batch_size if args.batch_size_test is None else args.batch_size_test
        ),
        "image_size": args.image_size,
        "max_epochs": args.epochs,
        "evaluate": args.evaluate,
        "arch": args.arch,
        "graph": args.graph,
        "gtn": args.gtn,
        "resume": args.resume,
        "num_classes": num_classes,
        "difficult_examples": False,
        "save_model_path": model_path,
        "workers": args.workers,
        "epoch_step": args.epoch_step,
        "lr": args.lr,
        "lr_graph": args.lrg,
        "lr_decay": args.lrd,
        "lr_scheduler": args.lrs,
        "lr_best": args.lrb,
        "device_ids": args.device_ids,
        "evaluate": True if args.evaluate else False,
    }

    engine = GraphMultiLabelEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer, scaler)


if __name__ == "__main__":
    main()
