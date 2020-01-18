from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
import sys

sys.path.append("/home/tbarton/Pointnet2_PyTorch/")
import etw2.etw_pytorch_utils as pt_utils
import pprint
import os.path as osp
import os
import argparse
import numpy as np
import h5py

torch.manual_seed(0)
from generate_fake_data import run
from pointnet2.models import Pointnet2SemMSG as Pointnet
from pointnet2.models.pointnet2_msg_sem import model_fn_decorator
from pointnet2.data import Indoor3DSemSeg

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-batch_size", type=int, default=2, help="Batch size [default: 32]"
)
parser.add_argument(
    "-num_points",
    type=int,
    default=4096,
    help="Number of points to train with [default: 4096]",
)
parser.add_argument(
    "-v",
    type=str,
    default="test",
    help="Number of points to train with [default: 4096]",
)
# parser.add_argument(
#     "-file_train",
#     type = str,
#     default = "/home/theresa/datav1balancedtrain.h5",
#     help = ""
#     )
# parser.add_argument(
#     "-file_test",
#     type = str,
#     default = "/home/theresa/datav1balancedtest.h5",
#     help = ""
#     )
parser.add_argument(
    "-weight_decay",
    type=float,
    default=0,
    help="L2 regularization coeff [default: 0.0]",
)
parser.add_argument(
    "-lr", type=float, default=1e-4, help="Initial learning rate [default: 1e-2]"
)
parser.add_argument(
    "-lr_decay",
    type=float,
    default=0.5,
    help="Learning rate decay gamma [default: 0.5]",
)
parser.add_argument(
    "-decay_step",
    type=float,
    default=2e5,
    help="Learning rate decay step [default: 20]",
)
parser.add_argument(
    "-bn_momentum",
    type=float,
    default=0.9,
    help="Initial batch norm momentum [default: 0.9]",
)
parser.add_argument(
    "-bn_decay",
    type=float,
    default=0.5,
    help="Batch norm momentum decay gamma [default: 0.5]",
)
parser.add_argument(
    "-checkpoint", type=bool, default=True, help="Checkpoint to start from"
)
parser.add_argument(
    "-epochs", type=int, default=1000, help="Number of epochs to train for"
)
parser.add_argument(
    "-run_name",
    type=str,
    default="sem_seg_run_1",
    help="Name for run in tensorboard_logger",
)
parser.add_argument("--visdom-port", type=int, default=8097)
parser.add_argument("--visdom", action="store_true")

lr_clip = 1e-5
bnm_clip = 1e-2

if __name__ == "__main__":
    args = parser.parse_args()
    c = args.v.split("/")
    c = c[-2] + c[-1]
    file_test = f"/home/theresa/p/{c}test.h5"
    file_train = f"/home/theresa/p/{c}train.h5"
    f = h5py.File(file_train)
    median_data = f["labels"][:]
    medians = []
    for data_file in median_data:
        medians.append(np.median(data_file))
    thresh = np.median(medians)
    test_set = Indoor3DSemSeg(args.num_points, file_test, train=False, thresh=thresh)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    train_set = Indoor3DSemSeg(args.num_points, file_train)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=2,
        shuffle=True,
    )

    model = Pointnet(num_classes=2, input_channels=0, use_xyz=True)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    lr_lbmd = lambda it: max(
        args.lr_decay ** (int(it * args.batch_size / args.decay_step)),
        lr_clip / args.lr,
    )
    bnm_lmbd = lambda it: max(
        args.bn_momentum
        * args.bn_decay ** (int(it * args.batch_size / args.decay_step)),
        bnm_clip,
    )

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    start_epoch = 1

    # load status from checkpoint
    checkpoint_name = f"checkpoints/pointnet2_{c}"
    best_name = f"checkpoints/pointnet2_{c}_best"
    # checkpoint_name = "checkpoints/" + "pointnet2_semseg.pth.tar"
    # checkpoint_name = "checkpoints/" + "pointnet2_semseg.pth.tar"

    if args.checkpoint:
        checkpoint_status = pt_utils.load_checkpoint(
            model, optimizer, filename=checkpoint_name.split(".")[0]
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd, last_epoch=it)
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bnm_lmbd, last_epoch=it
    )

    it = max(it, 0)  # for the initialize value of `trainer.train`
    weight = torch.tensor([.58, .42])
    model_fn = model_fn_decorator(nn.CrossEntropyLoss(weight=weight.cuda()))

    if args.visdom:
        viz = pt_utils.VisdomViz(port=args.visdom_port)
    else:
        viz = pt_utils.CmdLineViz()

    viz.text(pprint.pformat(vars(args)))

    evaluator = pt_utils.Eval(
        model,
        model_fn,results_folder=c
    )
    print(len(train_loader), len(test_loader))
    _ = evaluator.eval_epoch(train_loader, "train_guesses")
    _ = evaluator.eval_epoch(test_loader, "test_guesses")
