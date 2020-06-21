import argparse
import os
from typing import Optional

import numpy as np
import plac
import torch
from numpy.lib.format import open_memmap
from torch.nn import functional as F
from torch.utils.data import DataLoader

from MiDaS.models.midas_net import MidasNet
from Video3D.dataset import NumpyDataset, create_image_transform
from Video3D.io import write_video, read_video, VideoData
from mannequinchallenge.models import pix2pix_model
from utils.tools import TimerBlock


def inference_lasinger(video_data, model_path, logger: Optional[TimerBlock] = None, batch_size=8):
    close_logger_on_exit = False

    if logger is None:
        logger = TimerBlock("Inference with Depth Estimation Network")
        logger.__enter__()
        close_logger_on_exit = True

    model = MidasNet(model_path, non_negative=True)
    model = model.cuda()
    model.eval()
    logger.log("Loaded model weights from {}.".format(model_path))

    transform = create_image_transform(video_data.height, video_data.width)

    video_dataset = NumpyDataset(video_data.frames, transform)
    data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        num_frames_processed = 0

        for batch_i, batch in enumerate(data_loader):
            images = batch.cuda()
            depth = model(images)
            depth = F.interpolate(depth.unsqueeze(1), size=(video_data.height, video_data.width), mode='bilinear',
                                  align_corners=True)
            yield depth.detach().cpu().numpy()

            num_frames_processed += images.shape[0]

            logger.log("Generated {}/{} depth maps.\r".format(num_frames_processed, video_data.num_frames), end="")

    print()

    if close_logger_on_exit:
        logger.__exit__(None, None, None)


# TODO: Make inference code DRY.
def inference_li(video_data, model_path, logger: Optional[TimerBlock] = None, batch_size=8):
    close_logger_on_exit = False

    if logger is None:
        logger = TimerBlock("Inference with Depth Estimation Network")
        logger.__enter__()
        close_logger_on_exit = True

    opt = argparse.Namespace(input='single_view', mode='Ours_Bilinear',
                             checkpoints_dir='', name='', isTrain=True,
                             gpu_ids='0', lr=0.0004, lr_policy='step', lr_decay_epoch=8)
    model = pix2pix_model.Pix2PixModel(opt, _isTrain=True)
    state_dict = torch.load(model_path)

    if not next(iter(state_dict.keys())).startswith("module."):
        state_dict = {"module.{}".format(k): v for k, v in state_dict.items()}

    model.netG.load_state_dict(state_dict)
    model.switch_to_eval()

    logger.log("Loaded model weights from {}.".format(model_path))

    transform = create_image_transform(video_data.height, video_data.width, normalise=True)

    video_dataset = NumpyDataset(video_data.frames, transform)
    data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        num_frames_processed = 0

        for batch_i, batch in enumerate(data_loader):
            images = batch.cuda()
            images = images.to(torch.float32)
            depth, _ = model.netG(images)
            depth = F.interpolate(depth, size=(video_data.height, video_data.width), mode='bilinear',
                                  align_corners=True)
            yield depth.detach().cpu().numpy()

            num_frames_processed += images.shape[0]

            logger.log("Generated {}/{} depth maps.\r".format(num_frames_processed, video_data.num_frames), end="")

    print()

    if close_logger_on_exit:
        logger.__exit__(None, None, None)


def create_and_save_depth(inference_fn, video_data, depth_estimation_model_path, dnn_depth_map_path, logger, batch_size):
    try:
        depth_maps = open_memmap(
            filename=dnn_depth_map_path,
            dtype=np.float32,
            mode='w+',
            shape=(video_data.num_frames, 1, *video_data.shape)
        )

        depth_map_generator = inference_fn(video_data, depth_estimation_model_path, logger, batch_size=batch_size)

        for batch_i, depth_map in enumerate(depth_map_generator):
            batch_start_idx = batch_size * batch_i
            # Sometimes the last batch is a different size to the rest, so we need to use the actual batch size rather
            # than the specified one.
            current_batch_size = depth_map.shape[0]
            batch_end_idx = batch_start_idx + current_batch_size
            depth_maps[batch_start_idx:batch_end_idx] = depth_map

        depth_maps.flush()

        logger.log("Saved DNN depth maps to {}.".format(dnn_depth_map_path))

        return depth_maps
    except Exception:
        logger.log("\nError occurred during creation of depth maps - deleting {}.".format(dnn_depth_map_path))
        os.remove(dnn_depth_map_path)
        raise


@plac.annotations(
    video_path=plac.Annotation("The path to the input video.", kind="option", type=str, abbrev="i"),
    model_path=plac.Annotation("The path to the depth estimation model weights.", kind="option", type=str, abbrev="m"),
    video_output_path=plac.Annotation("The path to the write the output to.", kind="option", type=str, abbrev="o"),
    batch_size=plac.Annotation("The mini-batch size to use for the depth estimation network.", kind="option", type=int),
)
def main(video_path, model_path, video_output_path, batch_size=8):
    with TimerBlock("Load Video") as block:
        video_data = read_video(video_path, block)

    with TimerBlock("Depth Estimation") as block:
        tmp_dir = ".tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        depth_map_path = os.path.join(tmp_dir, "inference_depth_maps.npy")

        depth_maps = create_and_save_depth(inference_li, video_data, model_path, depth_map_path, block, batch_size=batch_size)
        depth_maps = (255 * (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min())).to(torch.uint8)
        write_video(VideoData(depth_maps, video_data.fps), video_output_path, block)


if __name__ == '__main__':
    plac.call(main)