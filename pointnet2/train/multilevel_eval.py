import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
import torch.nn as nn
from tensorboardX import SummaryWriter
import sys

sys.path.append("/home/tbarton/Pointnet2_PyTorch/")
import etw2.etw_pytorch_utils as pt_utils
import os
import argparse

torch.manual_seed(0)
from generate_fake_data import read_point_clouds
from pointnet2.models import Pointnet2SemMSG as Pointnet
from pointnet2.models.pointnet2_msg_sem import model_fn_decorator
from pointnet2.train.train_sem_seg import image_grid, load_data, eval_model
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

outpath = "model_output"
data_path = "data"
VERBOSE = False
lr_clip = 1e-5
bnm_clip = 1e-2
max_evalimages = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_print(s, of):
    with open(of, 'a') as f:
        f.write(f"{s}\n")
    print(s)


def loadConfigFile(exp_name):
    args = None
    with open(f"{outpath}/{exp_name}/config.txt") as f:
        for line in f:
            args = eval(line)

    assert args is not None, 'failed to load config'
    return args




def writeConfigFile(args):
    os.system(f'mkdir {outpath} > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name} > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/plots > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/points > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/models > /dev/null 2>&1')
    with open(f"{outpath}/{args.exp_name}/config.txt", "w") as f:
        f.write(f"{args}\n")


def run_train(exp_name, checkpoints, dataset, rd_seed, holdout_perc, batch_size,
              load_epochs=None, n_epochs=200, eval_frequency=20):
    random.seed(rd_seed)
    np.random.seed(rd_seed)
    torch.manual_seed(rd_seed)
    train_loader, test_loader = load_data(dataset, results_folder,
                                          holdout_perc, batch_size,
                                          exp_names[0])
    for net, load_epoch in zip(checkpoints, load_epochs):
        input_folder = f"{outpath}/{net}"
        checkpoint_name = lambda e: f"{input_folder}/models/eval_epoch_{e}.ckpt"
        best_checkpoint_name = lambda \
            e: f"{input_folder}/models/eval_epoch_{e}_best.ckpt"
    # writer = SummaryWriter(f"runs/{exp_name}") if \
    #     load_epoch is None else SummaryWriter(f"runs"
    #                                           f"/{exp_name + str(load_epoch)}")


        print('training ...')
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

        model = Pointnet(num_classes=2, input_channels=3, use_xyz=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

        if load_epoch is not None:
            file_name = checkpoint_name(load_epoch)
            loading_results = pt_utils.load_checkpoint(model, optimizer, file_name)
            if loading_results is None:
                raise IOError("Unable to create file {}".format(file_name))
            it, epoch, best_percentage = loading_results

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd,
                                         last_epoch=it)
        bnm_scheduler = pt_utils.BNMomentumScheduler(model, bn_lambda=bnm_lmbd,
                                                     last_epoch=it)

        it = max(it, 0)  # for the initialize value of `trainer.train`
        val_it = 0
        weight = torch.tensor([.58, .42])
        if args.one_class:
            loss_func = torch.nn.BCEWithLogitsLoss()
        else:
            loss_func = nn.CrossEntropyLoss(weight=weight.cuda())

        model_fn = model_fn_decorator(loss_func)

        for batch in train_loader:
            model.eval()
            if bnm_scheduler is not None:
                bnm_scheduler.step(it)


            preds, loss, eval_res = model_fn(model, batch,
                                             results_folder=results_folder)
            it += 1



            if test_loader is not None:
                val_loss, res = eval_model(model, model_fn, test_loader,
                                           epoch, results_folder, writer)

                is_best = val_loss < best_loss
                best_loss = min(best_loss, val_loss)


def eval_model(model, model_fn, d_loader, epoch, results_folder, writer):
    model.eval()

    eval_dict = {}
    total_loss = 0.0
    count = 1.0
    for i, data in enumerate(d_loader):
        preds, loss, eval_res = model_fn(model, data, eval=True, epoch=epoch,
                                         results_folder=results_folder)

        total_loss += loss.item()
        count += 1

        for k, v in eval_res.items():
            if v is not None:
                eval_dict[k] = eval_dict.get(k, 0) + v
        if i < max_evalimages:
            eval_dict["data"] = eval_dict.get("data", []) + [data]
            eval_dict["preds"] = eval_dict.get("preds", []) + [preds]

    return total_loss / count, eval_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arg parser")
    parser.add_argument('-en1', '--exp_name_1', help='name of experiment',
                        type=str)
    parser.add_argument('-en2', '--exp_name_2', help='name of experiment',
                        type=str)
    parser.add_argument('-mn', '--model_name', default=None,
                        help='name of the model used for evaluation, do not specify for model_name == exp_name',
                        type=str)
    parser.add_argument('-d', '--dataset', help='dataset to use', type=str)
    parser.add_argument("-batch_size", type=int, default=2,
                        help="Batch size [default: 32]")
    parser.add_argument("-num_points", type=int, default=4000,
                        help="Number of points to train with [default: 4096]", )
    parser.add_argument("-weight_decay", type=float, default=0,
                        help="L2 regularization coeff [default: 0.0]", )
    parser.add_argument("-lr", type=float, default=1e-4,
                        help="Initial learning rate [default: 1e-2]")
    parser.add_argument("-lr_decay", type=float, default=0.5,
                        help="Learning rate decay gamma [default: 0.5]", )
    parser.add_argument("-decay_step", type=float, default=2e5,
                        help="Learning rate decay step [default: 20]", )
    parser.add_argument("-bn_momentum", type=float, default=0.9,
                        help="Initial batch norm momentum [default: 0.9]", )
    parser.add_argument("-bn_decay", type=float, default=0.5,
                        help="Batch norm momentum decay gamma [default: 0.5]", )
    parser.add_argument("-epochs", type=int, default=1000,
                        help="Number of epochs to train for")
    parser.add_argument('-m', '--mode', default="load", type=str)
    parser.add_argument('-le1', '--load_epoch_1', default=None, type=int)
    parser.add_argument('-le2', '--load_epoch_2', default=None, type=int)

    parser.add_argument('-rd', '--rd_seed', default=42, type=int)
    parser.add_argument('-ho', '--holdout_perc', default=.1, type=float)
    parser.add_argument('-oc', '--one_class', default=True, type=bool)
    parser.add_argument('-ti', '--train_idx',
                        default="model_output/1class/train_idx.txt", type=str)
    parser.add_argument('-si', '--test_idx',
                        default="model_output/1class/test_idx.txt", type=str)

    args = parser.parse_args()
    if args.mode == "load":
        if not args.load_epoch:
            raise IOError
        en=[args.exp_name_1, args.exp_name_2]
        le = [args.load_epoch_1, args.load_epoch_2]
        run_train(
            en, args.dataset, args.rd_seed, args.holdout_perc,
            args.batch_size, le
        )
    # if args.mode == "train":
    #     writeConfigFile(args)
    #     run_train(
    #         args.exp_name, args.dataset, args.rd_seed, args.holdout_perc,
    #         args.batch_size, args.load_epoch
    #     )
