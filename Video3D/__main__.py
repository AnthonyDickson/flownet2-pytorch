import math
import os
import pickle
import warnings

import cv2
import numpy as np
import plac
import torch
from torch.nn import functional as F
from torchvision.transforms import Compose

from MiDaS.models.transforms import Resize
from Video3D.align_images import align_images
from Video3D.colmap_parsing import parse_cameras, parse_images, parse_points
from Video3D.dataset import OpticalFlowDatasetBuilder, wrap_MiDaS_transform, unwrap_MiDaS_transform, \
    create_image_transform
from Video3D.inference import inference
from Video3D.io import read_video, write_video
from Video3D.training import train
from models import FlowNet2
from utils.tools import TimerBlock


def parse_colmap_output(colmap_output_path, cache_path, colmap_cache_path, logger):
    try:
        with open(colmap_cache_path, 'rb') as f:
            camera, poses_by_image, points2d_by_image, points3d_by_id = pickle.load(f)

        logger.log("Loaded parsed COLMAP output from {}.".format(colmap_cache_path))
    except FileNotFoundError:
        cameras_txt = os.path.join(colmap_output_path, "cameras.txt")
        images_txt = os.path.join(colmap_output_path, "images.txt")
        points_3d_txt = os.path.join(colmap_output_path, "points3D.txt")

        camera = parse_cameras(cameras_txt)[0]
        logger.log("Parsed camera intrinsics.")

        poses_by_image, points2d_by_image = parse_images(images_txt)
        logger.log("Parsed camera poses and 2D points.")

        points3d_by_id = parse_points(points_3d_txt)
        logger.log("Parsed 3D points.")

        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)

        with open(colmap_cache_path, 'wb') as f:
            pickle.dump((camera, poses_by_image, points2d_by_image, points3d_by_id), f)

        logger.log("Saved parsed COLMAP output to {}.".format(colmap_cache_path))
    return camera, points2d_by_image, points3d_by_id, poses_by_image


def generate_sparse_depth_maps(camera, points2d_by_image, points3d_by_id, logger, sparse_depth_map_cache_path):
    try:
        sparse_depth_maps = np.load(sparse_depth_map_cache_path)
        logger.log("Loaded sparse depth maps from {}.".format(sparse_depth_map_cache_path))
    except IOError:
        sparse_depth_maps = []

        num_frames = len(points2d_by_image)

        for points in points2d_by_image.values():
            depth_map = np.zeros(shape=camera.shape, dtype=np.float32)

            for point in points:
                if point.point3d_id > -1 and point.x <= depth_map.shape[1] and point.y <= depth_map.shape[0]:
                    point3d = points3d_by_id[point.point3d_id]

                    depth_map[int(point.y), int(point.x)] = point3d.z

            sparse_depth_maps.append(depth_map)

            logger.log("Generated {}/{} sparse depth maps.\r".format(len(sparse_depth_maps), num_frames), end="")

        print()

        sparse_depth_maps = np.array(sparse_depth_maps, dtype=np.float32)
        np.save(sparse_depth_map_cache_path, sparse_depth_maps)
        logger.log("Saved sparse depth maps to {}.".format(sparse_depth_map_cache_path))

    return sparse_depth_maps


def calculate_global_scale_adjustment_factor(video_data, camera, points2d_by_image, points3d_by_id,
                                             depth_estimation_model_path, dnn_depth_map_cache_path,
                                             sparse_depth_map_cache_path, relative_depth_scale_cache_path, logger):
    try:
        with open(relative_depth_scale_cache_path, 'rb') as f:
            relative_depth_scale = pickle.load(f)

        logger.log("Loaded relative depth scale from {}".format(relative_depth_scale_cache_path))
    except FileNotFoundError:
        try:
            depth_maps = np.load(dnn_depth_map_cache_path)
            logger.log("Loaded DNN depth maps from {}.".format(dnn_depth_map_cache_path))
        except IOError:
            depth_maps = inference(video_data, depth_estimation_model_path, logger, batch_size=4).squeeze()

            np.save(dnn_depth_map_cache_path, depth_maps)
            logger.log("Saved DNN depth maps to {}.".format(dnn_depth_map_cache_path))

        sparse_depth_maps = generate_sparse_depth_maps(camera, points2d_by_image, points3d_by_id, logger,
                                                       sparse_depth_map_cache_path)

        zero_mask = sparse_depth_maps == 0

        relative_depth_scale = np.divide(depth_maps, sparse_depth_maps, dtype=np.float32)
        relative_depth_scale[zero_mask] = np.nan
        relative_depth_scale = np.nanmedian(relative_depth_scale, axis=[-2, -1])
        relative_depth_scale = np.mean(relative_depth_scale)

        logger.log("Calculated relative depth scale factor: {:.2f}.".format(relative_depth_scale))

        with open(relative_depth_scale_cache_path, 'wb') as f:
            pickle.dump(relative_depth_scale, f)

        logger.log("Saved relative depth scale factor to {}.".format(relative_depth_scale_cache_path))
    return relative_depth_scale


def sample_frame_pairs(num_frames):
    frame_pairs = {(i, i + 1) for i in range(num_frames - 1)}

    for l in range(math.floor(math.log2(num_frames - 1)) + 1):
        sl = {(i, i + 2 ** l) for i in range(num_frames - 2 ** l) if i % 2 ** (l - 1) == 0}
        frame_pairs = frame_pairs.union(sl)

    return frame_pairs


def create_optical_flow_dataset(video_data, camera, poses_by_image, optical_flow_model_path, dataset_cache_path,
                                logger):
    frame_pair_indexes = sample_frame_pairs(video_data.num_frames)
    logger.log("Sampled frame pairs")

    # TODO: Make this less hacky...
    # Simulate argparse object because I'm too lazy to implement it properly.
    class Object(object):
        pass

    args = Object()
    args.fp16 = False
    args.rgb_max = 255

    t = Compose(
        [
            wrap_MiDaS_transform,
            Resize(
                video_data.width,
                video_data.height,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=64,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            unwrap_MiDaS_transform
        ]
    )

    flow_net = FlowNet2(args).cuda()
    flow_net.load_state_dict(torch.load(optical_flow_model_path)["state_dict"])
    logger.log("Loaded optical flow model from {}.".format(optical_flow_model_path))
    os.makedirs(dataset_cache_path)
    num_filtered = 0

    def calculate_optical_flow(frame_i, frame_j):
        images = list(map(t, (frame_i, frame_j)))
        images = np.array(images).transpose((3, 0, 1, 2))
        images_tensor = torch.from_numpy(images).unsqueeze(0).to(torch.float32).cuda()

        optical_flow = flow_net(images_tensor)
        optical_flow = F.interpolate(optical_flow, size=(video_data.height, video_data.width),
                                     mode='bilinear', align_corners=True)
        optical_flow = optical_flow.squeeze().cpu().numpy()

        return optical_flow

    with torch.no_grad(), OpticalFlowDatasetBuilder(dataset_cache_path, camera) as dataset_builder:
        for pair_index, (i, j) in enumerate(frame_pair_indexes):
            frame_i = video_data.frames[i]
            frame_j = video_data.frames[j]
            frame_i_aligned = None
            frame_j_aligned = None

            try:
                frame_j_aligned = align_images(frame_i, frame_j)
            except ValueError:
                warnings.warn("Could not align frame #{} and #{}.".format(i, j))
                continue

            try:
                frame_i_aligned = align_images(video_data.frames[j], video_data.frames[i])
            except ValueError:
                warnings.warn("Could not align frame #{} and #{}.".format(j, i))
                continue

            optical_flow_forwards = calculate_optical_flow(frame_i, frame_j_aligned)
            optical_flow_backwards = calculate_optical_flow(frame_i_aligned, video_data.frames[j])

            delta = np.abs(optical_flow_forwards - optical_flow_backwards)
            valid_mask = (delta <= 1).astype(np.bool)
            # `valid_mask` is up to this point, indicating if each of the u and v components of each optical
            # flow vector are within 1px error.
            # However, we need it to indicate this on a per-pixel basis, so we combine the binary maps of the u
            # and v components to give us the validity of the optical flow at the given pixel.
            valid_mask = valid_mask[0, :, :] & valid_mask[1, :, :]
            # Ensure that valid mask is a CHW tensor to follow with PyTorch's conventions of dimension ordering.
            valid_mask = np.expand_dims(valid_mask, axis=0)

            # TODO: Check if `delta = np.sum(np.abs(a - b), axis=-1) <= 1` would do the same thing as above.
            should_keep_frame = np.mean(valid_mask) >= 0.8

            if should_keep_frame:
                dataset_builder.add(i, j, frame_i, frame_j, optical_flow_forwards, valid_mask, poses_by_image)
            else:
                num_filtered += 1

            logger.log("Processed {} frame pairs out of {} ({} kept, {} filtered out).\r"
                       .format(pair_index + 1, len(frame_pair_indexes), pair_index + 1 - num_filtered,
                               num_filtered), end="")
    print()
    logger.log("Saved dataset to {}.".format(dataset_cache_path))


@plac.annotations(
    colmap_output_path=plac.Annotation(
        "The path to the folder containing the text files `cameras.txt`, `images.txt` and `points3D.txt`."),
    video_path=plac.Annotation('The path to the source video file.', type=str,
                               kind='option', abbrev='i'),
    depth_estimation_model_path=plac.Annotation("The path to the pretrained MiDaS model weights.", type=str,
                                                kind="option", abbrev='d'),
    optical_flow_model_path=plac.Annotation("The path to the pretrained FlowNet2 model weights.", type=str,
                                            kind="option", abbrev='f'),
    cache_path=plac.Annotation("Where to save and load intermediate results to and from.", type=str, kind="option",
                               abbrev="c")
)
def main(colmap_output_path: str, video_path: str, depth_estimation_model_path: str = "model.pt",
         optical_flow_model_path: str = "FlowNet2_checkpoint.pth.tar", cache_path: str = ".cache"):
    # TODO: Refactor caching stuff into own class?
    cache_path = os.path.abspath(cache_path)
    video_name = os.path.basename(video_path)
    cache_path = os.path.join(cache_path, video_name)
    colmap_cache_path = os.path.join(cache_path, "parsed_colmap_output.pkl")
    sparse_depth_map_cache_path = os.path.join(cache_path, "sparse_depth_map.npy")
    dnn_depth_map_cache_path = os.path.join(cache_path, "dnn_depth_map.npy")
    relative_depth_scale_cache_path = os.path.join(cache_path, "relative_depth_scale.pkl")
    dataset_cache_path = os.path.join(cache_path, "dataset")
    optimised_dnn_weights_cache_path = os.path.join(cache_path, "optimised_depth_estimation_model.pt")
    optimised_dnn_checkpoint_cache_path = os.path.join(cache_path, "depth_estimation_model_checkpoint.pt")
    optimised_dnn_depth_map_cache_path = os.path.join(cache_path, "optimised_dnn_depth_map.npy")

    with TimerBlock("Load Video") as block:
        video_data = read_video(video_path, block)

    if not os.path.isdir(dataset_cache_path):
        with TimerBlock("Parse COLMAP Output") as block:
            camera, points2d_by_image, points3d_by_id, poses_by_image = parse_colmap_output(colmap_output_path,
                                                                                            cache_path,
                                                                                            colmap_cache_path, block)

        with TimerBlock("Calculate Relative Depth Scaling Factor") as block:
            relative_depth_scale = calculate_global_scale_adjustment_factor(video_data, camera,
                                                                            points2d_by_image, points3d_by_id,
                                                                            depth_estimation_model_path,
                                                                            dnn_depth_map_cache_path,
                                                                            sparse_depth_map_cache_path,
                                                                            relative_depth_scale_cache_path, block)

            for pose in poses_by_image.values():
                pose.t = relative_depth_scale * pose.t

            block.log("Scaled the translation component of the camera poses by the relative scale factor of {:.2f}"
                      .format(relative_depth_scale))

        with TimerBlock("Create Optical Flow Dataset") as block:
            create_optical_flow_dataset(video_data, camera, poses_by_image, optical_flow_model_path, dataset_cache_path,
                                        block)

        del camera, points2d_by_image, points3d_by_id, poses_by_image
    else:
        print("Found dataset at {}.".format(dataset_cache_path))

    if not os.path.isfile(optimised_dnn_weights_cache_path):
        with TimerBlock("Optimise Network") as block:
            train(depth_estimation_model_path, optimised_dnn_weights_cache_path, optimised_dnn_checkpoint_cache_path,
                  dataset_cache_path, block)

    with TimerBlock("Generate Refined Depth Maps") as block:
        depth_maps = inference(video_data, depth_estimation_model_path, block, scale_output=True, batch_size=4)
        write_video(depth_maps, video_data, "output-before.avi", block)
        block.log("Generated depth maps with un-optimised network.")

        depth_maps = inference(video_data, optimised_dnn_weights_cache_path, block, scale_output=True, batch_size=4)
        write_video(depth_maps, video_data, "output-after.avi", block)
        block.log("Generated depth maps with optimised network.")

        np.save(optimised_dnn_depth_map_cache_path, depth_maps)
        block.log("Saved optimised depth maps to {}.".format(optimised_dnn_depth_map_cache_path))


if __name__ == '__main__':
    plac.call(main)
