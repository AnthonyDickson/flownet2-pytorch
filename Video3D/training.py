import argparse
import warnings

import numpy as np
import plac
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from MiDaS.models.midas_net import MidasNet
from Video3D.colmap_io import Camera
from Video3D.dataset import OpticalFlowDataset, create_image_transform
from mannequinchallenge.models import pix2pix_model
from utils.tools import TimerBlock


def point3d_to_point2d(x):
    return x[:2] / x[2]


class RunningAverage:
    def __init__(self):
        self.count = 0
        self.sum = 0

    def update(self, value, count=1):
        if value == float('inf'):
            warnings.warn("{} got a invalid value of {}.".format(self.__class__.__name__, value))
        else:
            self.sum += value
            self.count += count

    @property
    def value(self):
        try:
            return self.sum / self.count
        except ZeroDivisionError:
            return float('nan')


@plac.annotations(
    model_path=plac.Annotation("The path to the pretrained model weights.", type=str, kind="option", abbrev="m"),
    best_weights_path=plac.Annotation("The path to the save the best model weights.", type=str, kind="option",
                                      abbrev="b"),
    checkpoint_path=plac.Annotation("The path to the save the model checkpoints.", type=str, kind="option", abbrev="c"),
    dataset_path=plac.Annotation("The path to the frame pair and optical flow dataset.", type=str, kind="option",
                                 abbrev="d"),
    num_epochs=plac.Annotation("The number of epochs to train the network for.", type=int, kind="option"),
    batch_size=plac.Annotation("The mini-batch size.", type=int, kind="option"),
    lr=plac.Annotation("The learning rate.", type=float, kind="option"),
    balancing_coefficient=plac.Annotation("The weighting of the disparity loss relative to the spatial loss.",
                                          type=float, kind="option"),
)
def train_lasinger(model_path, best_weights_path, checkpoint_path, dataset_path, logger=None,
                   num_epochs=20, batch_size=8, lr=0.0004, balancing_coefficient=0.1):
    close_logger_on_exit = False

    if logger is None:
        logger = TimerBlock("Optimise Depth Estimation Network")
        logger.__enter__()
        close_logger_on_exit = True

    transform = create_image_transform(height=384, width=384)
    logger.log("Created image transform.")

    flow_dataset = OpticalFlowDataset(dataset_path, transform)
    logger.log("Created optical flow dataset.")

    data_loader = DataLoader(flow_dataset, batch_size=batch_size, shuffle=True)
    logger.log("Created data loader for optical flow dataset.")

    # TODO: Does setting non_negative=True affect training like it did in my other experiments (outputs all going to zero).
    model = MidasNet(model_path, non_negative=True)
    model = model.cuda()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    logger.log("Loaded depth estimation model.")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.1)

    K = flow_dataset.metadata.camera.get_matrix()
    K = Camera.to_homogeneous_matrix(K)
    K = torch.from_numpy(K).to(torch.float32).cuda()

    K_inverse = flow_dataset.metadata.camera.get_inverse_matrix()
    K_inverse = Camera.to_homogeneous_matrix(K_inverse)
    K_inverse = torch.from_numpy(K_inverse).to(torch.float32).cuda()

    focal_length = flow_dataset.metadata.camera.focal_length

    batch = flow_dataset[0]
    optical_flow_mask = batch[-1]
    _, height, width = optical_flow_mask.shape

    x = torch.from_numpy(np.array(np.meshgrid(range(width), range(height)), dtype=np.float32))
    x_ = torch.ones(size=(4, *x.shape[1:]), dtype=torch.float32)
    x_[:2, :, :] = x

    x = x.cuda()
    x_ = x_.cuda()

    best_loss = float('inf')

    for epoch in range(num_epochs):
        logger.log("Epoch {}/{}".format(epoch + 1, num_epochs))
        epoch_loss = RunningAverage()
        epoch_progress = 0

        for batch in data_loader:
            frame_i, _, R_i, t_i, R_j, t_j, optical_flow, valid_mask = batch
            batch_size, _, height, width = optical_flow.shape

            optimiser.zero_grad()

            depth_i = model(frame_i.cuda())
            depth_i = F.interpolate(depth_i.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=True)

            # TODO: Tidy up loss function and refactor into function or something?
            p_ij = torch.zeros((batch_size, 2, height, width)).cuda()
            disparity_ij = torch.zeros((batch_size, 1, height, width)).cuda()
            R_i, t_i, R_j, t_j, optical_flow, valid_mask = R_i.cuda(), t_i.cuda(), R_j.cuda(), t_j.cuda(), optical_flow.cuda(), valid_mask.cuda()

            f_ij = optical_flow + torch.stack(batch_size * [x], dim=0)

            # TODO: Vectorise this...
            for batch_i in range(batch_size):
                # The tensordot function will essentially take K^-1 * x_ for every pixel.
                c_i = depth_i[batch_i] * torch.tensordot(K_inverse, x_, dims=1)
                c_ij = torch.tensordot(torch.transpose(R_j[batch_i], 0, 1),
                                       torch.tensordot(R_i[batch_i], c_i, dims=1) + (
                                               t_i[batch_i] - t_j[batch_i]).unsqueeze(-1), dims=1)

                c_ij_homogeneous = torch.ones(size=(4, *c_ij.shape[1:]), dtype=c_ij.dtype, device=c_ij.device)
                c_ij_homogeneous[:-1] = c_ij

                p_ij[batch_i] = point3d_to_point2d(torch.tensordot(K, c_ij_homogeneous, dims=1))

                disparity_ij[batch_i] = 1.0 / c_i[-1] - 1.0 / c_ij[-1]

            stacked_valid_mask = torch.cat([valid_mask] * 2, dim=1)
            l_spatial = torch.sqrt(torch.sum(torch.pow((f_ij - p_ij), 2)[stacked_valid_mask]))
            l_disparity = focal_length * torch.sum(torch.abs(disparity_ij[valid_mask]))

            num_valid_pixels = valid_mask.to(torch.float32).sum()
            weighted_loss = l_spatial + balancing_coefficient * l_disparity

            loss = 1.0 / num_valid_pixels * weighted_loss

            loss.backward()
            optimiser.step()
            lr_scheduler.step()

            epoch_progress += batch_size
            epoch_loss.update(weighted_loss.item(), num_valid_pixels.item())
            logger.log(
                "Epoch Progress: {:04d}/{:04d} - Loss {:.4f} (Epoch Avg.: {:.4f})          \r"
                    .format(epoch_progress, len(flow_dataset), loss.item(), epoch_loss.value)
                , end=""
            )

        print()

        torch.save(model.state_dict(), checkpoint_path)

        if epoch_loss.value < best_loss:
            best_loss = epoch_loss.value
            torch.save(model.state_dict(), best_weights_path)
            logger.log("Saved best model weights to {}.".format(best_weights_path))

    if close_logger_on_exit:
        logger.__exit__(None, None, None)


# TODO: Make training code DRY.
@plac.annotations(
    batch_size=plac.Annotation("Mini-batch size.", kind="option")
)
def train_li(model_path, checkpoint_path, best_weights_path, dataset_path, logger=None,
             num_epochs=20, batch_size=8, lr=0.0004, balancing_coefficient=0.1):
    close_logger_on_exit = False

    if logger is None:
        logger = TimerBlock("Optimise Depth Estimation Network")
        logger.__enter__()
        close_logger_on_exit = True

    opt = argparse.Namespace(input='single_view', mode='Ours_Bilinear',
                             checkpoints_dir='', name='',
                             gpu_ids='0', isTrain=True, lr=lr, lr_policy='step', lr_decay_epoch=5)
    model = pix2pix_model.Pix2PixModel(opt, _isTrain=True)
    state_dict = torch.load(model_path)

    if not next(iter(state_dict.keys())).startswith("module."):
        state_dict = {"module.{}".format(k): v for k, v in state_dict.items()}

    model.netG.load_state_dict(state_dict)
    model.switch_to_train()
    logger.log("Loaded model weights from {}.".format(model_path))

    transform = create_image_transform(height=384, width=384, normalise=True)
    logger.log("Created image transform.")

    flow_dataset = OpticalFlowDataset(dataset_path, transform)
    logger.log("Created optical flow dataset.")

    data_loader = DataLoader(flow_dataset, batch_size=batch_size, shuffle=True)
    logger.log("Created data loader for optical flow dataset.")

    K = flow_dataset.metadata.camera.get_matrix()
    K = Camera.to_homogeneous_matrix(K)
    K = torch.from_numpy(K).to(torch.float32).cuda()

    K_inverse = flow_dataset.metadata.camera.get_inverse_matrix()
    K_inverse = Camera.to_homogeneous_matrix(K_inverse)
    K_inverse = torch.from_numpy(K_inverse).to(torch.float32).cuda()

    focal_length = flow_dataset.metadata.camera.focal_length

    batch = flow_dataset[0]
    optical_flow_mask = batch[-1]
    _, height, width = optical_flow_mask.shape

    x = torch.from_numpy(np.array(np.meshgrid(range(width), range(height - 1, -1, -1)), dtype=np.float32))
    x_ = torch.ones(size=(4, *x.shape[1:]), dtype=torch.float32)
    x_[:2, :, :] = x

    x = x.cuda()
    x_ = x_.cuda()

    best_loss = float('inf')

    for epoch in range(num_epochs):
        logger.log("Epoch {}/{}".format(epoch + 1, num_epochs))
        epoch_loss = RunningAverage()
        epoch_progress = 0

        for batch in data_loader:
            frame_i, _, R_i, t_i, R_j, t_j, optical_flow, valid_mask = batch
            batch_size, _, height, width = optical_flow.shape

            model.optimizer_G.zero_grad()

            depth_i, _ = model.netG(frame_i.cuda())
            depth_i = F.interpolate(depth_i, size=(height, width), mode="bilinear", align_corners=True)

            # TODO: Tidy up loss function and refactor into function or something?
            p_ij = torch.zeros((batch_size, 2, height, width)).cuda()
            disparity_ij = torch.zeros((batch_size, 1, height, width)).cuda()
            R_i, t_i, R_j, t_j, optical_flow, valid_mask = R_i.cuda(), t_i.cuda(), R_j.cuda(), t_j.cuda(), optical_flow.cuda(), valid_mask.cuda()

            f_ij = optical_flow + torch.stack(batch_size * [x], dim=0)

            # TODO: Vectorise this...
            for batch_i in range(batch_size):
                # The tensordot function will essentially take K^-1 * x_ for every pixel.
                c_i = depth_i[batch_i] * torch.tensordot(K_inverse, x_, dims=1)
                c_ij = torch.tensordot(torch.transpose(R_j[batch_i], 0, 1),
                                       torch.tensordot(R_i[batch_i], c_i, dims=1) + (
                                               t_i[batch_i] - t_j[batch_i]).unsqueeze(-1), dims=1)

                c_ij_homogeneous = torch.ones(size=(4, *c_ij.shape[1:]), dtype=c_ij.dtype, device=c_ij.device)
                c_ij_homogeneous[:-1] = c_ij

                p_ij[batch_i] = point3d_to_point2d(torch.tensordot(K, c_ij_homogeneous, dims=1))

                disparity_ij[batch_i] = 1.0 / c_i[-1] - 1.0 / c_ij[-1]

            stacked_valid_mask = torch.cat([valid_mask] * 2, dim=1)
            l_spatial = torch.sqrt(torch.sum(torch.pow((f_ij - p_ij), 2)[stacked_valid_mask]))
            l_disparity = focal_length * torch.sum(torch.abs(disparity_ij[valid_mask]))

            num_valid_pixels = valid_mask.to(torch.float32).sum()
            weighted_loss = l_spatial + balancing_coefficient * l_disparity

            loss = 1.0 / num_valid_pixels * weighted_loss

            loss.backward()
            model.optimizer_G.step()
            model.lr_scheduler.step()

            epoch_progress += batch_size
            epoch_loss.update(weighted_loss.item(), num_valid_pixels.item())
            logger.log(
                "Epoch Progress: {:04d}/{:04d} - Loss {:.4f} (Epoch Avg.: {:.4f})          \r"
                    .format(epoch_progress, len(flow_dataset), loss.item(), epoch_loss.value)
                , end=""
            )

        print()

        torch.save(model.netG.state_dict(), checkpoint_path)

        if epoch_loss.value < best_loss:
            best_loss = epoch_loss.value
            torch.save(model.netG.state_dict(), best_weights_path)
            logger.log("Saved best model weights to {}.".format(best_weights_path))

    if close_logger_on_exit:
        logger.__exit__(None, None, None)


if __name__ == '__main__':
    plac.call(train_li)
