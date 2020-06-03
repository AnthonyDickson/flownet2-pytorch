import os
import pickle
import warnings

import cv2
import numpy as np
import plac
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from MiDaS.models.midas_net import MidasNet
from MiDaS.models.transforms import Resize, PrepareForNet, NormalizeImage
from Video3D.colmap_parsing import Camera
from Video3D.dataset import OpticalFlowDataset
from utils.tools import TimerBlock


def wrap_MiDaS_transform(x):
    return {"image": x}


def unwrap_MiDaS_transform(x):
    return x["image"]


@plac.annotations(
    model_path=plac.Annotation("The path to the pretrained model weights", type=str, kind="option", abbrev="m"),
    dataset_path=plac.Annotation("The path to the frame pair and optical flow dataset.", type=str, kind="option",
                                 abbrev="d"),
    cache_dir=plac.Annotation("Where to save intermediate results to (such as the trained model weights).", type=str, kind="option", abbrev="c"),
)
def train(model_path, dataset_path, cache_dir, num_epochs=20, batch_size=4, lr=0.0004,
          balancing_coefficient=0.1):
    trained_model_weights_cache_path = os.path.join(cache_dir, "optimised_depth_estimation_model.pt")
    trained_model_checkpoint_cache_path = os.path.join(cache_dir, "depth_estimation_model_checkpoint.pt")

    with TimerBlock("Optimise Network") as block:
        transform = Compose(
            [
                wrap_MiDaS_transform,
                Resize(
                    384,
                    384,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
                unwrap_MiDaS_transform
            ]
        )
        block.log("Created image transform.")

        flow_dataset = OpticalFlowDataset(dataset_path, transform)
        block.log("Created optical flow dataset.")

        data_loader = DataLoader(flow_dataset, batch_size=batch_size, shuffle=True)
        block.log("Created data loader for optical flow dataset.")

        model = MidasNet(model_path, non_negative=False)
        model = model.cuda()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        block.log("Loaded depth estimation model.")

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

        best_loss = float('inf')

        for epoch in range(num_epochs):
            block.log("Epoch {}/{}".format(epoch + 1, num_epochs))
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
                    c_ij = torch.tensordot(torch.transpose(R_j[batch_i], 0, 1), torch.tensordot(R_i[batch_i], c_i, dims=1) + (t_i[batch_i] - t_j[batch_i]).unsqueeze(-1), dims=1)

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
                block.log("Epoch Progress: {:04d}/{:04d} - Loss {:.4f} (Epoch Avg.: {:.4f})\r".format(epoch_progress, len(flow_dataset), loss.item(), epoch_loss.value), end="")

            print()

            torch.save(model.state_dict(), trained_model_checkpoint_cache_path)

            if epoch_loss.value < best_loss:
                best_loss = epoch_loss.value
                torch.save(model.state_dict(), trained_model_weights_cache_path)
                block.log("Saved best model weights to {}.".format(trained_model_weights_cache_path))


if __name__ == '__main__':
    plac.call(train)
