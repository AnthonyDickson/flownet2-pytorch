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

        def homogeneous_translation_vector(v):
            v_ = torch.ones((v.shape[0] + 1, v.shape[1]), device=v.device)
            v_[:-1, :] = v

            return v_

        def pi(x):
            return x[:-1, :] / x[-1, :]

        for epoch in range(num_epochs):
            for batch in data_loader:
                frame_i, _, R_i, t_i, R_j, t_j, optical_flow, valid_mask = batch
                channels, height, width = optical_flow.shape[:3]

                optimiser.zero_grad()

                depth_i = model(frame_i.cuda())
                depth_i = F.interpolate(depth_i.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=True)

                f_ij = torch.zeros((channels, height, width, 2)).cuda()
                p_ij = torch.zeros((channels, height, width, 2)).cuda()
                disparity_ij = torch.zeros((channels, height, width)).cuda()
                R_i, t_i, R_j, t_j, optical_flow, valid_mask = R_i.cuda(), t_i.cuda(), R_j.cuda(), t_j.cuda(), optical_flow.cuda(), valid_mask.cuda()

                # TODO: Vectorise this...
                for channel in range(channels):
                    for row in range(height):
                        for col in range(width):
                            x = torch.tensor([row, col], dtype=optical_flow.dtype).cuda()
                            x_ = torch.tensor([[row, col, 1.0, 1.0]], dtype=torch.float32).T.cuda()

                            f_ij[channel, row, col] = optical_flow[channel, row, col] + x

                            c_i = depth_i[channel, 0, row, col] * torch.matmul(K_inverse, x_)
                            c_ij = torch.matmul(R_j[channel].T,
                                                (torch.matmul(R_i[channel], c_i) + t_i[channel] - t_j[channel]))

                            p_ij[channel, row, col] = pi(
                                torch.matmul(K, homogeneous_translation_vector(c_ij))).flatten()

                            disparity_ij[channel, row, col] = 1.0 / c_i[-1] - 1.0 / c_ij[-1]

                l_spatial = torch.dist(p_ij[valid_mask], f_ij[valid_mask], p=2)
                l_disparity = focal_length * torch.sum(torch.abs(disparity_ij[valid_mask]))

                loss = 1.0 / valid_mask.to(torch.float32).sum() * (l_spatial + balancing_coefficient * l_disparity)

                loss.backward()

            # TODO: Log progress and loss


if __name__ == '__main__':
    plac.call(main)
