import os
import pickle

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
    colmap_output_cache_path=plac.Annotation("The path to the processed COLMAP output.", type=str, kind="option",
                                             abbrev="c"),
)
def main(model_path, dataset_path, colmap_output_cache_path, num_epochs=20, batch_size=4, lr=0.0004,
         balancing_coefficient=0.1):
    with TimerBlock("Setup") as block:
        for filename in os.listdir(dataset_path):
            if filename.endswith(".png"):
                img = cv2.imread(os.path.join(dataset_path, filename))
                height, width, _ = img.shape

                break

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

        model = MidasNet(model_path, non_negative=True)
        model = model.cuda()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        block.log("Loaded depth estimation model.")

        optimiser = torch.optim.Adam(model.parameters(), lr=lr)

        K = flow_dataset.metadata.camera.get_matrix()
        K = Camera.to_homogeneous_matrix(K)
        K = torch.from_numpy(K).to(torch.float32).cuda()

        K_inverse = flow_dataset.metadata.camera.get_inverse_matrix()
        K_inverse = Camera.to_homogeneous_matrix(K_inverse)
        K_inverse = torch.from_numpy(K_inverse).to(torch.float32).cuda()

        focal_length = flow_dataset.metadata.camera.focal_length

        def point3d_to_point2d(x):
            return x[:2] / x[2]

        for epoch in range(num_epochs):
            for batch in data_loader:
                frame_i, _, R_i, t_i, R_j, t_j, optical_flow, valid_mask = batch
                batch_size, _, height, width = optical_flow.shape

                optimiser.zero_grad()

                depth_i = model(frame_i.cuda())
                depth_i = F.interpolate(depth_i.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=True)

                f_ij = torch.zeros((batch_size, 2, height, width)).cuda()
                p_ij = torch.zeros((batch_size, 2, height, width)).cuda()
                disparity_ij = torch.zeros((batch_size, 1, height, width)).cuda()
                R_i, t_i, R_j, t_j, optical_flow, valid_mask = R_i.cuda(), t_i.cuda(), R_j.cuda(), t_j.cuda(), optical_flow.cuda(), valid_mask.cuda()

                # TODO: Vectorise this...
                for batch_i in range(batch_size):
                    x = torch.from_numpy(np.array(np.meshgrid(range(width), range(height)), dtype=np.float32))
                    x_ = torch.ones(size=(4, *x.shape[1:]), dtype=torch.float32)
                    x_[:2, :, :] = x

                    x = x.cuda()
                    x_ = x_.cuda()

                    f_ij[batch_i] = optical_flow[batch_i] + x
                    # The tensordot function will essentially take K^-1 * x_ for every pixel.
                    c_i = depth_i[batch_i] * torch.tensordot(K_inverse, x_, dims=1)
                    c_ij = torch.tensordot(R_j[batch_i].T, torch.tensordot(R_i[batch_i], c_i, dims=1) + (t_i[batch_i] - t_j[batch_i]).unsqueeze(2), dims=1)

                    c_ij_homogeneous = torch.ones(size=(4, *c_ij.shape[1:]), dtype=c_ij.dtype, device=c_ij.device)
                    c_ij_homogeneous[:-1] = c_ij

                    p_ij[batch_i] = point3d_to_point2d(torch.tensordot(K, c_ij_homogeneous, dims=1))

                    disparity_ij[batch_i] = 1.0 / c_i[-1] - 1.0 / c_ij[-1]

                stacked_valid_mask = torch.cat([valid_mask] * 2, dim=1)
                l_spatial = torch.sqrt(torch.sum(torch.pow((f_ij - p_ij), 2)[stacked_valid_mask]))
                l_disparity = focal_length * torch.sum(torch.abs(disparity_ij[valid_mask]))

                loss = 1.0 / valid_mask.to(torch.float32).sum() * (l_spatial + balancing_coefficient * l_disparity)

                loss.backward()

            # TODO: Log progress and loss


if __name__ == '__main__':
    plac.call(main)
