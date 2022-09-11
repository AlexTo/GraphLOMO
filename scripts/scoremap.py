import argparse
import wsmol

from torch import nn
from scripts.wsol_train import Trainer
from data_loaders import get_data_loader
from inference import CAMComputer
from os.path import join as ospj
from config import _ARCHITECTURE_NAMES, _DATASET_NAMES, _METHOD_NAMES, box_v2_metric, check_dependency, configure_data_paths, configure_log, configure_mask_root, configure_pretrained_path, configure_reporter, configure_scoremap_output_paths, get_wsol_method, str2bool


def get_config():
    parser = argparse.ArgumentParser()

    # Util
    parser.add_argument('--seed', type=int)
    parser.add_argument('--experiment_name', type=str, default='test_case')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    # Data
    parser.add_argument('--dataset_name', type=str, default='CUB',
                        choices=_DATASET_NAMES)
    parser.add_argument('--data_root', metavar='/PATH/TO/DATASET',
                        default='dataset/',
                        help='path to dataset images')
    parser.add_argument('--metadata_root', type=str, default='metadata/')
    parser.add_argument('--mask_root', metavar='/PATH/TO/MASKS',
                        default='dataset/',
                        help='path to masks')
    parser.add_argument('--proxy_training_set', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='Efficient hyper_parameter search with a proxy '
                             'training set.')
    parser.add_argument('--num_val_sample_per_class', type=int, default=0,
                        help='Number of full_supervision validation sample per '
                             'class. 0 means "use all available samples".')
    # Setting
    parser.add_argument('--architecture', default='resnet50',
                        choices=_ARCHITECTURE_NAMES,
                        help='model architecture: ' +
                             ' | '.join(_ARCHITECTURE_NAMES) +
                             ' (default: resnet50)')
    parser.add_argument('--pretrained', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='Use pre_trained model.')
    parser.add_argument('--check_point', type=str, default=None)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--cam_curve_interval', type=float, default=.001,
                        help='CAM curve interval')
    parser.add_argument('--resize_size', type=int, default=512,
                        help='input resize size')
    parser.add_argument('--crop_size', type=int, default=448,
                        help='input crop size')
    parser.add_argument('--multi_contour_eval', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--multi_iou_eval', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--iou_threshold_list', nargs='+',
                        type=int, default=[30, 50, 70])
    parser.add_argument('--eval_checkpoint_type', type=str, default='last',
                        choices=('best', 'last'))
    parser.add_argument('--checkpoint_interval', type=int, default=-1,
                        help='Save checkpoint every n epoch, set to -1 to disable. Works independently of eval_checkpoint_type')
    parser.add_argument('--box_v2_metric', type=str2bool, nargs='?',
                        const=True, default=True)

    # Common hyperparameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Mini-batch size (default: 256), this is the total'
                             'batch size of all GPUs on the current node when'
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--large_feature_map', type=str2bool, nargs='?',
                        const=True, default=False)

    # Method-specific hyperparameters
    parser.add_argument('--wsol_method', type=str, default='cam',
                        choices=_METHOD_NAMES)
    parser.add_argument('--has_grid_size', type=int, default=4)
    parser.add_argument('--has_drop_rate', type=float, default=0.5)
    parser.add_argument('--acol_threshold', type=float, default=0.7)
    parser.add_argument('--spg_threshold_1h', type=float, default=0.7,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_1l', type=float, default=0.01,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2h', type=float, default=0.5,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2l', type=float, default=0.05,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3h', type=float, default=0.7,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3l', type=float, default=0.1,
                        help='SPG threshold')
    parser.add_argument('--adl_drop_rate', type=float, default=0.75,
                        help='ADL dropout rate')
    parser.add_argument('--adl_threshold', type=float, default=0.9,
                        help='ADL gamma, threshold ratio '
                             'to maximum value of attention map')
    parser.add_argument('--cutmix_beta', type=float, default=1.0,
                        help='CutMix beta')
    parser.add_argument('--cutmix_prob', type=float, default=1.0,
                        help='CutMix Mixing Probability')
    parser.add_argument('--embedding', default='model/embedding/coco_glove_word2vec_80x300.pkl',
                        type=str, metavar='EMB', help='path to embedding (default: glove)')
    parser.add_argument('--emb_features', default=300, type=int, metavar='EMB',
                        help='embedding length (default: 300)')
    parser.add_argument('--gtn', '--transformer', dest='gtn', action='store_true',
                        help='use graph transformer networks')
    parser.add_argument('--adj_dd_threshold', default=0.4, type=float, metavar='ADJ_T',
                        help='Data-driven adj threshod (default: 0.4')
    parser.add_argument('--adj_files', default=['model/topology/coco_adj.pkl'], type=str, nargs='+',
                        help='Adj files (default: [coco_adj.pkl])')

    args = parser.parse_args()

    args.log_folder = ospj('train_log', args.experiment_name)
    box_v2_metric(args)

    args.wsol_method = get_wsol_method(args.wsol_method)
    args.data_paths = configure_data_paths(args)
    args.metadata_root = ospj(args.metadata_root, args.dataset_name)
    args.mask_root = configure_mask_root(args)
    args.scoremap_paths = configure_scoremap_output_paths(args)
    args.reporter, args.reporter_log_root = configure_reporter(args)
    args.pretrained_path = ospj(
        args.log_folder, args.check_point) if args.check_point else None
    args.spg_thresholds = ((args.spg_threshold_1h, args.spg_threshold_1l),
                           (args.spg_threshold_2h, args.spg_threshold_2l),
                           (args.spg_threshold_3h, args.spg_threshold_3l))

    return args


def main():
    args = get_config()

    classif_type = Trainer._CLASSIF_PROBLEM_MAPPING[args.dataset_name]
    num_classes = Trainer._NUM_CLASSES_MAPPING[args.dataset_name]

    loaders = get_data_loader(
        data_roots=args.data_paths,
        metadata_root=args.metadata_root,
        batch_size=args.batch_size,
        workers=args.workers,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        proxy_training_set=args.proxy_training_set,
        num_classes=num_classes,
        classif_type=classif_type,
        num_val_sample_per_class=args.num_val_sample_per_class)

    model = wsmol.__dict__[args.architecture](
        dataset_name=args.dataset_name,
        wsol_method=args.wsol_method,
        pretrained=args.pretrained,
        num_classes=num_classes,
        classif_type=classif_type,
        large_feature_map=args.large_feature_map,
        pretrained_path=args.pretrained_path,
        adl_drop_rate=args.adl_drop_rate,
        adl_drop_threshold=args.adl_threshold,
        acol_drop_threshold=args.acol_threshold,
        adj_dd_threshold=args.adj_dd_threshold,
        adj_files=args.adj_files,
        embedding=args.embedding,
        emb_features=args.emb_features,
        gtn=args.gtn)

    model = model.cuda()
    cam_computer = CAMComputer(model=model,
                               loader=loaders[args.split],
                               metadata_root=ospj(
                                   args.metadata_root, args.split),
                               mask_root=args.mask_root,
                               iou_threshold_list=args.iou_threshold_list,
                               dataset_name=args.dataset_name,
                               split=args.split,
                               cam_curve_interval=args.cam_curve_interval,
                               multi_contour_eval=args.multi_contour_eval,
                               log_folder=args.log_folder,
                               classif_type=classif_type,
                               crop_size=args.crop_size)

    cam_computer.compute_and_evaluate_cams(evaluate_cam=False)


if __name__ == "__main__":
    main()
