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
from datetime import datetime

sys.path.append("/home/tbarton/Pointnet2_PyTorch/")
import etw2.etw_pytorch_utils as pt_utils
import pprint
import os.path as osp
import os
import argparse

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
    "-fst",
    type=str,
    default="test",
    help="Number of points to train with [default: 4096]",
)
parser.add_argument(
    "-snd",
    type=str,
    default="test",
    help="Number of points to train with [default: 4096]",
)
parser.add_argument(
    "-trd",
    type=str,
    default="test",
    help="Number of points to train with [default: 4096]",
)
parser.add_argument(
    "-data_train",
    type=str,
    default="/home/theresa/p/datav2balancedtrain.h5",
    help="Number of points to train with [default: 4096]",
)
parser.add_argument(
    "-data_test",
    type=str,
    default="/home/theresa/p/datav2balancedtest.h5",
    help="Number of points to train with [default: 4096]",
)
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

def evaluate_step(args, input_name, input_path, checkpoint_name,
                  best_checkpoint_name, output_name, file_test = None, file_train = None):
    # input_name is the directory name in gsd that contains the files that the network must process.
    # input_path is the full path to the files
    outpath= f"/home/theresa/Pointnet2_PyTorch/pointnet2/generated_shapes_debug/"
    if file_test and file_train:
        pass
    else:
        file_test = f"generated_shapes_debug/{input_name}test.h5"
        file_train = f"generated_shapes_debug/{input_name}train.h5"
        # if not os.path.exists(file_test):
        # make train and test files
        run(input_name,
            inpath=input_path,
            outpath=outpath)
    test_set = Indoor3DSemSeg(args.num_points, file_test, train=False)
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

    _ = pt_utils.load_checkpoint(
        model, optimizer, filename=checkpoint_name.split(".")[0]
    )

    weight = torch.tensor([.58, .42])
    model_fn = model_fn_decorator(nn.CrossEntropyLoss(weight=weight.cuda()))

    evaluator = pt_utils.Eval(
        model,
        model_fn,
        results_folder=output_name
    )
    print(len(train_loader), len(test_loader))
    _ = evaluator.eval_epoch(test_loader, "test_guesses1")
    _ = evaluator.eval_epoch(test_loader, "test_guesses2")
    _ = evaluator.eval_epoch(test_loader, "test_guesses3")
    # TODO resample points
    pts = {}
    import numpy as np
    for i in range(1,4):
        path = outpath + "00152020-01-09T08/" + f"test_guesses{i}"
        for p in os.path.listdir(path):
            csv = np.genfromtext(path + "/" + p, dtype=np.float64, delimiter=",")



if __name__ == "__main__":
    # python3 train/multilevel_eval.py -snd 00152020-01-06T08 -trd 00132020-01-06T13
    tm = datetime.now().isoformat()
    tm = tm[:-13]
    args = parser.parse_args()
    # load initial data and first (fst)
    # checkpoint_name = "checkpoints/coarsetocoarse/" + "pointnet2_semseg.pth.tar"
    # best_name = "checkpoints/coarsetocoarse/" + "pointnet2_semseg_best.pth.tar"
    # evaluate_step(args, "", "", output_name = f"fst_{args.fst}",
    #               checkpoint_name=checkpoint_name, best_checkpoint_name=best_name, file_test=args.data_test,
    #               file_train = args.data_train)
    checkpoint_name = f"checkpoints/pointnet2_{args.snd}.pth.tar"
    best_name = f"checkpoints/pointnet2_{args.snd}best.pth.tar"
    input_path = f"/home/theresa/Pointnet2_PyTorch/pointnet2/generated_shapes_debug/fst_{args.fst}/traintest_guesses/"
    input_path = f"/home/theresa/Pointnet2_PyTorch/pointnet2/generated_shapes_debug/fst_{args.fst}+resampled/"
    #moving onto second model
    evaluate_step(args, f"fst_{args.fst}",
                  input_path,
                  checkpoint_name=checkpoint_name, best_checkpoint_name=best_name,
                  output_name = f"snd_{args.snd}")

    # third and final model
    # checkpoint_name = f"checkpoints/pointnet2_{args.trd}.pth.tar"
    # best_name = f"checkpoints/pointnet2_{args.trd}best.pth.tar"
    #
    # evaluate_step(args, f"snd_{args.snd}",
    #               f"/home/theresa/Pointnet2_PyTorch/pointnet2/"
    #               f"generated_shapes_debug/snd_{args.snd}/traintest_guesses/",
    #               checkpoint_name=checkpoint_name, best_checkpoint_name=best_name,
    #               output_name = f"trd_{args.trd}")
