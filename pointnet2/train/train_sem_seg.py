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
from pointnet2.data.Indoor3DSemSegLoader import Uvloader
from pointnet2.utils import loss_functions
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

def image_grid(inpoints, scores):
    # Create a figure to contain the plot.
    scores = scores.cpu()
    points, groundtruth = (inpoints[0].cpu(), inpoints[1].cpu())
    fig = plt.figure(figsize=(10, 10))
    plt.grid(False)
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    red = lambda x: [x,0,1-x]
    colors = lambda x:[red(y.item()) for y in x]
    ax2.scatter(points[0,:,0], -1*points[0,:,2], points[0,:,1], c = colors(groundtruth[0]), s=.5)
    ax1.scatter(points[0,:,0], -1*points[0,:,2], points[0,:,1], c = colors(scores[0].cpu()), s=.5)
    ax2.scatter(points[0,:,0], -1*points[0,:,2], points[0,:,1], c = colors(groundtruth[0]), s=.5)
    ax3.scatter(points[1,:,0], -1*points[1,:,2], points[1,:,1], c = colors(scores[1].cpu()), s=.5)
    ax4.scatter(points[1,:,0], -1*points[1,:,2], points[1,:,1], c = colors(groundtruth[1]), s=.5)
    def axplt(ax, X, Y, Z):

        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten()
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten()
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten()
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
    axplt(ax1, points[0,:,0], -1*points[0,:,2], points[0,:,1])
    axplt(ax2, points[0,:,0], -1*points[0,:,2], points[0,:,1])
    axplt(ax3, points[1,:,0], -1*points[1,:,2], points[1,:,1])
    axplt(ax4, points[1,:,0], -1*points[1,:,2], points[1,:,1])

    return fig

def writeConfigFile(args):
    os.system(f'mkdir {outpath} > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name} > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/plots > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/points > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/models > /dev/null 2>&1')
    with open(f"{outpath}/{args.exp_name}/config.txt", "w") as f:
        f.write(f"{args}\n")

def load_data(dataset, results_folder, holdout_perc, batch_size, exp_name):
    pc, dists, idx_gt = read_point_clouds(f"{data_path}/{dataset}/")
    # print(pc[0])
    pc_t, dists_t, idx_t = [], [], []
    for i in range(len(pc)):
        cloud = pc[i]
        new_cloud = []
        new_dists = []
        for d, j in zip(dists[i], cloud):
            if j[0] > 0:
                new_cloud.append(j)
                new_dists.append(d)
        pc_t.append(new_cloud)
        dists_t.append(new_dists)
    max_pts = 1000000
    min_pts = 4000
    exclude_idx = [38]
    for i in range(len(pc_t)):
        if len(pc_t[i]) < min_pts:
            exclude_idx.append(i)
        elif len(pc_t[i]) < max_pts:
            max_pts = len(pc_t[i])
    pc_t = [pc[:max_pts] for i, pc in enumerate(pc_t) if i not in exclude_idx]
    dists_t = [dc[:max_pts] for i, dc in enumerate(dists_t) if
               i not in exclude_idx]
    idx_t = [ic for i, ic in enumerate(idx_gt) if i not in exclude_idx]

    points_labels_index = list(zip(pc_t, dists_t, idx_t))
    if args.train_idx and args.test_idx:
        pc, dists, idx_gt = zip(*points_labels_index)
        ti = list(np.genfromtxt(args.train_idx, dtype=np.int32, delimiter=","))
        si = list(np.genfromtxt(args.test_idx, dtype=np.int32, delimiter=","))
        train = [pli for pli in points_labels_index if pli[2] in ti]
        test = [pli for pli in points_labels_index if pli[2] in si]
        train_points, train_dist, train_idx = zip(*train)
        test_points, test_dist, test_idx = zip(*test)

    else:
        random.shuffle(points_labels_index)
        train_num = int((len(pc_t) * (1.0 - holdout_perc)) + .5)
        train = points_labels_index[:train_num]
        test = points_labels_index[train_num:]
        train_points, train_dist, train_idx = zip(*train)
        test_points, test_dist, test_idx = zip(*test)

    train_thresh = np.median(np.array(train_dist), axis=1, keepdims=True)
    log_print(f"train median{np.median(train_thresh, axis=0)}", f"{outpath}/"
                                                                f"{exp_name}/log.txt")

    # Use train_thresh for both to avoid cheating
    train_thresh = np.broadcast_to(np.median(train_thresh, axis=0), np.array(
        train_dist).shape)
    test_thresh = np.broadcast_to(np.median(train_thresh, axis=0), np.array(
        test_dist).shape)

    train_labels = np.where(train_dist > train_thresh, 1., 0.)
    test_labels = np.where(test_dist > test_thresh, 1., 0.)

    train_above = np.average(np.sum(train_labels, axis=1) / train_labels.shape[
        1], axis=0)
    log_print(f"% train above{train_above}", f"{outpath}/{exp_name}/log.txt")

    test_above = np.average(np.sum(test_labels, axis=1) / test_labels.shape[1],
                            axis=0)
    log_print(f"% test above{test_above}", f"{outpath}/{exp_name}/log.txt")

    log_print(f"Training indices:", f"{outpath}/{exp_name}/log.txt")
    for ind in train_idx:
        log_print(f"{ind}", f"{outpath}/{exp_name}/log.txt")

    log_print(f"Validation indices:", f"{outpath}/{exp_name}/log.txt")
    for ind in test_idx:
        log_print(f"{ind}", f"{outpath}/{exp_name}/log.txt")
    train_loader = DataLoader(Uvloader(train_points, train_labels,
                                       train_dist, train_idx),
                              batch_size=2, shuffle=True)
    test_loader = DataLoader(Uvloader(test_points, test_labels,
                                      test_dist, test_idx),
                             batch_size=2, shuffle=True)

    log_print(f"Training size: {len(train_idx)}", f"{outpath}"
                                                  f"/{exp_name}/log.txt")
    log_print(f"Validation size: {len(test_idx)}", f"{outpath}"
                                                   f"/{exp_name}/log.txt")
    return train_loader, test_loader

def run_train(exp_name, dataset, rd_seed, holdout_perc, batch_size,
              load_epoch=None, n_epochs=200, eval_frequency=20):

    random.seed(rd_seed)
    np.random.seed(rd_seed)
    torch.manual_seed(rd_seed)

    results_folder = f"{outpath}/{exp_name}"
    checkpoint_name = lambda e: f"{results_folder}/models/eval_epoch_{e}.ckpt"
    best_checkpoint_name = lambda e: f"{results_folder}/models/eval_epoch_{e}_best.ckpt"
    writer = SummaryWriter(f"runs/{exp_name }") if \
        load_epoch is None else SummaryWriter(f"runs"
                                            f"/{exp_name + str(load_epoch)}")
    train_loader, test_loader = load_data(dataset, results_folder,
                                          holdout_perc, batch_size, exp_name)


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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if load_epoch is not None:
        file_name = checkpoint_name(load_epoch)
        loading_results = pt_utils.load_checkpoint(model, optimizer, file_name)
        if loading_results is None:
            raise IOError("Unable to create file {}".format(file_name))
        it, epoch, best_percentage = loading_results

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd, last_epoch=it)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bn_lambda=bnm_lmbd, last_epoch=it)

    it = max(it, 0)  # for the initialize value of `trainer.train`
    val_it = 0
    weight = torch.tensor([.58, .42])
    if args.loss_function == "one_class":
        loss_func = torch.nn.BCEWithLogitsLoss()
    elif args.loss_function == "pr":
        PRLoss = loss_functions.PRLoss(device)
        loss_func = PRLoss.forward_loss()
    else:
        loss_func = nn.CrossEntropyLoss(weight=weight.cuda())


    model_fn = model_fn_decorator(loss_func)

    for epoch in range(n_epochs):
        for batch in train_loader:
            if args.mode == "load":
                model.eval()
            elif args.mode == "train":
                model.train()

            if bnm_scheduler is not None:
                bnm_scheduler.step(it)

            optimizer.zero_grad()

            preds, loss, eval_res = model_fn(model, batch, results_folder=results_folder)
            if args.mode != "load":

                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step(it)
            it += 1

            if (it % eval_frequency) == 0:

                if test_loader is not None:
                    val_loss, res = eval_model(model, model_fn, test_loader,
                                               epoch, results_folder, writer)

                    is_best = val_loss < best_loss
                    best_loss = min(best_loss, val_loss)
                    pt_utils.save_checkpoint(
                        pt_utils.checkpoint_state(
                            model, optimizer, val_loss, epoch, it
                        ),
                        is_best,
                        filename=checkpoint_name(epoch),
                        bestname=best_checkpoint_name(epoch),
                    )
                    val_it = val_it + 1
                    writer.add_scalar("ValidationLoss", val_loss, val_it)
                    for i in range(max_evalimages):
                        figure = image_grid(res["data"][i], res["preds"][i])
                        writer.add_figure("validation_image", figure, val_it)
                    writer.add_scalar("ValidationPrecision", res["precision"],
                                      val_it)
                    writer.add_scalar("ValidationRecall", res["recall"],
                                      val_it)

            writer.add_scalar("TrainingLoss", loss.item(), it)
            figure = image_grid(batch, preds)
            writer.add_figure("training_image", figure, it)


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
    parser.add_argument('-en', '--exp_name', help='name of experiment', type=str)
    parser.add_argument('-mn', '--model_name', default=None,help='name of the model used for evaluation, do not specify for model_name == exp_name', type=str)
    parser.add_argument('-d', '--dataset', help='dataset to use', type=str)
    parser.add_argument("-batch_size", type=int, default=2, help="Batch size [default: 32]")
    parser.add_argument("-num_points",type=int,default=4000,help="Number of points to train with [default: 4096]",)
    parser.add_argument("-weight_decay",type=float,default=0,help="L2 regularization coeff [default: 0.0]",)
    parser.add_argument("-lr", type=float, default=1e-4, help="Initial learning rate [default: 1e-2]")
    parser.add_argument("-lr_decay",type=float,default=0.5,help="Learning rate decay gamma [default: 0.5]",)
    parser.add_argument("-decay_step",type=float,default=2e5,help="Learning rate decay step [default: 20]",)
    parser.add_argument("-bn_momentum",type=float,default=0.9,help="Initial batch norm momentum [default: 0.9]",)
    parser.add_argument("-bn_decay",type=float,default=0.5,help="Batch norm momentum decay gamma [default: 0.5]",)
    parser.add_argument("-epochs", type=int, default=1000, help="Number of epochs to train for")
    parser.add_argument('-m', '--mode', default="train", type=str)
    parser.add_argument('-le', '--load_epoch', default=None, type=int)
    parser.add_argument('-rd', '--rd_seed', default=42, type=int)
    parser.add_argument('-ho', '--holdout_perc', default = .1, type = float)
    parser.add_argument('-lf', '--loss_function', default="pr", type = str)
    parser.add_argument('-ti', '--train_idx',
                        default="model_output/1class/train_idx.txt", type =str)
    parser.add_argument('-si', '--test_idx',
                        default="model_output/1class/test_idx.txt", type=str)


    args = parser.parse_args()
    if args.mode == "load":
        if not args.load_epoch:
            raise IOError
        run_train(
            args.exp_name, args.dataset, args.rd_seed, args.holdout_perc,
            args.batch_size, args.load_epoch
        )
    if args.mode == "train":
        writeConfigFile(args)
        run_train(
            args.exp_name, args.dataset, args.rd_seed, args.holdout_perc, args.batch_size, args.load_epoch
)
