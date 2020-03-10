from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
from collections import namedtuple
torch.manual_seed(0)
from pointnet2.utils.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG

Verbose = True
low_dist = 0.03
import os

def plot_points(output_folder, points, scores, dists, index):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    for batch, cloud in enumerate(points):
        score = scores[batch]
        dist = dists[batch]
        fname = output_folder + "/{}.csv".format(str(index[batch].item()))
        with open(fname, 'w+') as f:
            for point_index in range(cloud.shape[0]):
                f.write(f"{cloud[point_index, 0]},"
                        f"{cloud[point_index, 1]},"
                        f"{cloud[point_index, 2]},"
                        f"{cloud[point_index, 3]},"
                        f"{cloud[point_index, 4]},"
                        f"{cloud[point_index, 5]},"
                        f"{score[point_index].item()},"
                        f"{dist[point_index].item()}\n" )

def isfinite(x):
    not_inf = ((x + 1) != x)
    not_nan = (x == x)
    return not_inf & not_nan

def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])

    def model_fn(model, data, epoch=0, eval=False, pfx="", results_folder="",
                 one_class=True):


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.set_grad_enabled(not eval):
            inputs, labels, dists, index = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)
            if one_class:
                preds = preds.squeeze(2)
                loss = criterion(preds.view(-1), labels.view(-1))
                preds_probabilities = torch.sigmoid(preds)
                # classes is 0 if on seam, 1 if off seam
                classes = torch.where(
                    preds_probabilities>torch.Tensor([0.5]).to(device),
                    torch.ones_like(preds_probabilities).to(device),
                    torch.zeros_like(preds_probabilities).to(device))
            else:
                loss = criterion(preds.view(labels.numel(), -1),
                                 labels.view(-1))

                preds_probabilities = torch.nn.functional.softmax(preds, -1)
                _, classes = torch.max(preds_probabilities, -1)
            acc = (classes == labels).float().sum() / labels.numel()
            inputs = inputs.cpu().numpy()
            labels = labels.cpu().numpy()
            index = index.numpy()

            # Plotting
            gt_folder_name = "eval_target" if eval else "target"
            pred_folder_name = "eval_preds" if eval else "preds"

            if (eval or (epoch % 10 == 0)) and Verbose:
                plot_points(f"{results_folder}/points/{gt_folder_name}",
                            inputs, labels, dists, index)
                plot_points(f"{results_folder}/points/{pred_folder_name}",
                            inputs, preds_probabilities, dists, index)
        results_dict = {"acc": acc.item(), "loss": loss.item()}
        if eval:
            calculate_pa(classes, dists, results_dict)

        if Verbose:
            print("loss:", loss.item())
            print("acc", acc.item())

        return ModelReturn(preds_probabilities, loss, results_dict)

    return model_fn

def calculate_pa(classes, dists, results_dict):
    results_dict["precision"] = 0.
    results_dict["recall"] = 0.
    for batch in range(classes.shape[0]):
        res = [dists[batch, i].item() for i, val in enumerate(classes[
                                                                  batch].cpu(

        ).numpy()) if val == 0]
        cls = [classes[batch].cpu().numpy()[i].item() for i, val in
               enumerate(dists[batch].numpy()) if val <= low_dist]
        cls_cor = [c for c in cls if c == 0]
        results_dict["precision"] += 0. if len(res) == 0 else sum(
            res) / float( len(res)) / float(classes.shape[0])
        results_dict["recall"] += 0. if len(cls) == 0 else len(
            cls_cor) / float(
            len(cls)) / float(classes.shape[0])
class Pointnet2MSG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propagation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature

    """

    def __init__(self, num_classes, input_channels=6, use_xyz=True):
        super(Pointnet2MSG, self).__init__()

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=use_xyz,
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.FC_layer = (
            pt_utils.Seq(128)
            .conv1d(128, bn=True)
            # .dropout()
            .conv1d(1, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            # print(l_xyz[i].shape)
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
            # print(l_features[i - 1].shape)
        # print(l_features[0].shape)
        return self.FC_layer(l_features[0]).transpose(1, 2).contiguous()


if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    import torch.optim as optim

    B = 2
    N = 32
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model = Pointnet2MSG(3, input_channels=3)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    print("Testing with xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for index in range(5):
        optimizer.zero_grad()
        preds, loss, acc = model_fn(model, (inputs, labels))
        # print(preds.shape)
        # print(inputs.shape)
        loss.backward()
        print(loss.data[0])
        optimizer.step()

    # with use_xyz=False
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model = Pointnet2MSG(3, input_channels=3, use_xyz=False)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    print("Testing without xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.data[0])
        optimizer.step()
