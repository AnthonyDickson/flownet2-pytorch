import argparse
import math
import os
import pickle
import subprocess
import warnings
from collections import deque
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import plac
import torch
from numpy.lib.format import open_memmap
from torch.nn import functional as F
from torchvision.transforms import Compose

from MiDaS.models.transforms import Resize
from Video3D.align_images import align_images
from Video3D.colmap_io import read_model
from Video3D.dataset import OpticalFlowDatasetBuilder, wrap_MiDaS_transform, unwrap_MiDaS_transform
from Video3D.inference import inference_li, inference_lasinger, create_and_save_depth
from Video3D.io import read_video
from Video3D.training import train_lasinger, train_li
from models import FlowNet2
from utils.tools import TimerBlock


@plac.annotations(
    video_path=plac.Annotation(
        'The path to the source video file.',
        type=str, kind='option', abbrev='i'
    ),
    checkpoint_path=plac.Annotation(
        "The path to the pretrained model weights for the depth estimation model and FlowNet2.",
        type=str, kind="option"
    ),
    workspace_path=plac.Annotation(
        "Where to save and load intermediate results to and from.",
        type=str, kind="option"
    ),
    depth_estimation_model=plac.Annotation(
        "Which depth estimation model to use.",
        type=str, kind="option", choices=["li", "lasinger"]
    ),
    batch_size=plac.Annotation(
        "The batch size to use for the depth estimation network. Decrease this if you are running out of VRAM",
        type=int, kind="option"
    )
)
def main(video_path: str, checkpoint_path: str = "checkpoints/", workspace_path: str = "workspace/",
         depth_estimation_model="li", batch_size: int = 8):
    print(dict(
        video_path=video_path,
        checkpoint_path=checkpoint_path,
        workspace_path=workspace_path,
        depth_estimation_model=depth_estimation_model,
        batch_size=batch_size
    ))

    assert os.path.isfile(video_path), "Could not open video located at {}.".format(video_path)

    video_name = Path(video_path).stem
    workspace_path = os.path.join(os.path.abspath(workspace_path), video_name)

    if not os.path.isdir(workspace_path):
        os.makedirs(workspace_path)

    frame_data_path = os.path.join(workspace_path, "frames")
    colmap_workspace_path = os.path.join(workspace_path, "colmap")
    optical_flow_model_path = os.path.join(checkpoint_path, "FlowNet2_checkpoint.pth.tar")
    optimised_depth_estimation_model_path = os.path.join(workspace_path, "optimised_depth_estimation_model.pt")
    optimised_dnn_checkpoint_path = os.path.join(workspace_path, "depth_estimation_model_checkpoint.pt")

    dnn_depth_map_path = os.path.join(workspace_path, "dnn_depth_map.npy")
    optimised_dnn_depth_map_path = os.path.join(workspace_path, "optimised_dnn_depth_map.npy")
    sparse_depth_map_path = os.path.join(workspace_path, "sparse_depth_map.npy")
    relative_depth_scale_path = os.path.join(workspace_path, "relative_depth_scale.pkl")

    dataset_path = os.path.join(workspace_path, "dataset")

    sample_video_path = os.path.join(workspace_path, "before_after_comparison.avi")

    if depth_estimation_model == "li":
        train_fn = train_li
        inference_fn = inference_li

        depth_estimation_model_path = os.path.join(
            checkpoint_path, "test_local", "best_depth_Ours_Bilinear_inc_3_net_G.pth"
        )
    elif depth_estimation_model == "lasinger":
        train_fn = train_lasinger
        inference_fn = inference_lasinger

        depth_estimation_model_path = os.path.join(
            checkpoint_path, "model.pt"
        )
    else:
        raise RuntimeError("Unsupported depth estimation model: {}.".format(depth_estimation_model))

    with TimerBlock("Load Video") as block:
        video_data = read_video(video_path, block)

    with TimerBlock("Create Optical Flow Dataset") as block:
        if os.path.isdir(dataset_path):
            block.log("Found dataset at {}.".format(dataset_path))
        else:
            create_optical_flow_dataset(video_data, frame_data_path, colmap_workspace_path, sparse_depth_map_path,
                                        dnn_depth_map_path, relative_depth_scale_path, dataset_path,
                                        depth_estimation_model_path, optical_flow_model_path, block, inference_fn,
                                        batch_size)

    with TimerBlock("Optimise Depth Estimation Network") as block:
        if os.path.isfile(optimised_depth_estimation_model_path):
            block.log("Found optimised network weights at {}.".format(optimised_depth_estimation_model_path))
        else:
            train_fn(depth_estimation_model_path, optimised_depth_estimation_model_path, optimised_dnn_checkpoint_path,
                     dataset_path, block, batch_size=batch_size)

    if not os.path.isfile(dnn_depth_map_path):
        with TimerBlock("Generate Depth Maps") as block:
            create_and_save_depth(inference_fn, video_data, depth_estimation_model_path, dnn_depth_map_path,
                                  logger=block, batch_size=batch_size)
            block.log("Generated depth maps with pretrained network.")
            block.log("Saved depth maps to {}.".format(dnn_depth_map_path))

    if not os.path.isfile(optimised_dnn_depth_map_path):
        with TimerBlock("Generate Optimised Depth Maps") as block:
            create_and_save_depth(inference_fn, video_data, optimised_depth_estimation_model_path,
                                  optimised_dnn_depth_map_path, logger=block, batch_size=batch_size)
            block.log("Generated depth maps with optimised network.")
            block.log("Saved depth maps to {}.".format(optimised_dnn_depth_map_path))

    with TimerBlock("Generate Sample Video") as block:
        generate_sample_video(video_path, dnn_depth_map_path, optimised_dnn_depth_map_path, sample_video_path, block)


def create_optical_flow_dataset(video_data, frame_data_path, colmap_workspace_path, sparse_depth_map_path,
                                dnn_depth_map_path, relative_depth_scale_path, dataset_path,
                                depth_estimation_model_path, optical_flow_model_path, logger, inference_fn, batch_size):
    run_colmap(video_data, frame_data_path, colmap_workspace_path, logger)
    camera, images_by_id, points3d_by_id = get_colmap_output(colmap_workspace_path, logger)

    relative_depth_scale = calculate_global_scale_adjustment_factor(video_data, camera, images_by_id,
                                                                    points3d_by_id, depth_estimation_model_path,
                                                                    dnn_depth_map_path,
                                                                    sparse_depth_map_path,
                                                                    relative_depth_scale_path, logger,
                                                                    inference_fn, batch_size)

    for image in images_by_id.values():
        image.camera_pose.t = relative_depth_scale * image.camera_pose.t

    logger.log("Scaled the translation component of the camera poses by the relative scale factor of {:.2f}"
               .format(relative_depth_scale))

    frame_pair_indexes = sample_frame_pairs(video_data.num_frames)
    logger.log("Sampled frame pairs")

    args = argparse.Namespace(fp16=False, rgb_max=255)

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

    os.makedirs(dataset_path)
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

    with torch.no_grad(), OpticalFlowDatasetBuilder(dataset_path, camera) as dataset_builder:
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
                dataset_builder.add(i, j, frame_i, frame_j, optical_flow_forwards, valid_mask, images_by_id)
            else:
                num_filtered += 1

            logger.log("Processed {} frame pairs out of {} ({} kept, {} filtered out).\r"
                       .format(pair_index + 1, len(frame_pair_indexes), pair_index + 1 - num_filtered,
                               num_filtered), end="")
    print()
    logger.log("Saved dataset to {}.".format(dataset_path))


def run_colmap(video_data, frame_data_path, colmap_workspace_path, logger):
    # TODO: Generate masked video with Detectron2.

    if not os.path.exists(frame_data_path):
        os.makedirs(frame_data_path)
        logger.log("Created folder for frame data at {}.".format(frame_data_path))

        for frame_i, frame in enumerate(video_data):
            frame_filename = "{:04d}.png".format(frame_i + 1)

            # If the video data is in the RGB format (as opposed to the BGR format), then it must be converted to BGR
            # before being written to disk since that is the format OpenCV uses.
            if video_data.is_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame_path = os.path.join(frame_data_path, frame_filename)
            cv2.imwrite(frame_path, frame)
            logger.log("Wrote frame {:,d}/{:,d} to {}.\r".format(frame_i + 1, len(video_data), frame_path), end="")

        print()

        logger.log("Wrote frame data for COLMAP to {}.".format(frame_data_path))
    else:
        logger.log("Found frame data at {}.".format(frame_data_path))

    if not os.path.exists(colmap_workspace_path):
        os.makedirs(colmap_workspace_path)
        logger.log("Created workspace folder at {}.".format(colmap_workspace_path))

        logger.log("Running COLMAP reconstruction. This may take a while...")
        colmap_process = subprocess.run([
            'colmap', 'automatic_reconstructor',
             '--image_path', frame_data_path,
             '--workspace_path', colmap_workspace_path,
             '--single_camera', '1',
             '--quality', 'low',
             '--data_type', 'video',
             '--camera_model', 'SIMPLE_PINHOLE',
             # TODO: Specify masks for dynamic objects
         ])

        if colmap_process.returncode != 0:
            raise RuntimeError("COLMAP exited with the non-zero return code {}.".format(colmap_process.returncode))
        else:
            logger.log("COLMAP finished processing the video.")

    else:
        logger.log("Found COLMAP reconstruction workspace folder at {}.".format(colmap_workspace_path))


def get_colmap_output(colmap_workspace_path, logger):
    cameras, images, points3D = read_model(
        os.path.join(colmap_workspace_path, "sparse/0"),
        ".bin"
    )
    logger.log("Loaded raw COLMAP output from {}.".format(colmap_workspace_path))

    # Assume there is only one camera. Sometimes COLMAP can give multiple cameras I think?
    camera = list(cameras.values())[0]

    if len(cameras) > 1:
        warnings.warn("More than one camera was found for the COLMAP output located at {}. "
                      "Expected only one camera.".format(colmap_workspace_path))

    return camera, images, points3D


def generate_sparse_depth_maps(camera, images_by_id, points3d_by_id, logger, sparse_depth_map_path):
    if os.path.isfile(sparse_depth_map_path):
        sparse_depth_maps = np.load(sparse_depth_map_path, mmap_mode='r')
        logger.log("Loaded sparse depth maps from {}.".format(sparse_depth_map_path))
    else:
        try:
            num_frames = len(images_by_id)

            sparse_depth_maps = open_memmap(
                filename=sparse_depth_map_path,
                dtype=np.float32,
                mode='w+',
                shape=(num_frames, 1, *camera.shape)
            )
            height, width = camera.shape

            for image_i, image in enumerate(images_by_id.values()):
                for point in image.points2D:
                    if point.point3d_id > -1 and point.x <= width and point.y <= height:
                        point3d = points3d_by_id[point.point3d_id]
                        # Have to subtract one from image ids since they are one-indexed to avoid index out of bounds
                        # and off by one errors.
                        sparse_depth_maps[image.id - 1, 0, int(point.y), int(point.x)] = point3d.z

                logger.log("Generated {:,d}/{:,d} sparse depth maps.\r".format(image_i + 1, num_frames), end="")

            print()

            sparse_depth_maps.flush()
            logger.log("Saved sparse depth maps to {}.".format(sparse_depth_map_path))
        except Exception:
            logger.log("\nError occurred during creation of sparse depth maps - deleting {}.".format(sparse_depth_map_path))
            os.remove(sparse_depth_map_path)
            raise

    return sparse_depth_maps


def calculate_global_scale_adjustment_factor(video_data, camera, images_by_id, points3d_by_id,
                                             depth_estimation_model_path, dnn_depth_map_path,
                                             sparse_depth_map_path, relative_depth_scale_path, logger,
                                             inference_fn, batch_size):
    try:
        with open(relative_depth_scale_path, 'rb') as f:
            relative_depth_scale = pickle.load(f)

        logger.log("Loaded relative depth scale from {}".format(relative_depth_scale_path))
    except FileNotFoundError:
        try:
            depth_maps = np.load(dnn_depth_map_path)
            logger.log("Loaded DNN depth maps from {}.".format(dnn_depth_map_path))
        except IOError:
            depth_maps = create_and_save_depth(inference_fn, video_data, depth_estimation_model_path,
                                               dnn_depth_map_path, logger, batch_size)

            logger.log("Saved DNN depth maps to {}.".format(dnn_depth_map_path))

        sparse_depth_maps = generate_sparse_depth_maps(camera, images_by_id, points3d_by_id, logger,
                                                       sparse_depth_map_path)

        zero_mask = sparse_depth_maps == 0

        relative_depth_scale = np.divide(depth_maps, sparse_depth_maps, dtype=np.float32)
        relative_depth_scale[zero_mask] = np.nan
        relative_depth_scale = np.nanmedian(relative_depth_scale, axis=[-2, -1])
        relative_depth_scale = np.nanmean(relative_depth_scale)

        logger.log("Calculated relative depth scale factor: {:.2f}.".format(relative_depth_scale))

        with open(relative_depth_scale_path, 'wb') as f:
            pickle.dump(relative_depth_scale, f)

        logger.log("Saved relative depth scale factor to {}.".format(relative_depth_scale_path))
    return relative_depth_scale


def sample_frame_pairs(num_frames):
    frame_pairs = {(i, i + 1) for i in range(num_frames - 1)}

    for l in range(math.floor(math.log2(num_frames - 1)) + 1):
        sl = {(i, i + 2 ** l) for i in range(num_frames - 2 ** l) if i % 2 ** (l - 1) == 0}
        frame_pairs = frame_pairs.union(sl)

    return frame_pairs


def generate_sample_video(video_path, dnn_depth_map_path, optimised_dnn_depth_map_path, sample_video_path, logger):
    def load_depth_map(path):
        x = np.load(path, mmap_mode='r')

        def transform(depth_maps):
            scaled = (255 * (depth_maps - x.min()) / (x.max() - x.min())).astype(np.uint8)
            return np.concatenate(3 * [scaled], axis=1)

        return x, transform

    if os.path.isfile(sample_video_path):
        logger.log("Found sample video at {}.".format(sample_video_path))
        return

    video_data = read_video(video_path, logger, convert_to_rgb=False)
    video_data = video_data.to_nchw()

    depth_maps, depth_maps_transform = load_depth_map(dnn_depth_map_path)
    logger.log("Loaded depth maps generated by original model from {}.".format(dnn_depth_map_path))

    optimised_depth_maps, optimised_depth_maps_transform = load_depth_map(optimised_dnn_depth_map_path)
    logger.log("Loaded optimised depth maps from {}.".format(optimised_dnn_depth_map_path))

    assert depth_maps.shape == optimised_depth_maps.shape, \
        "Shape of the depth maps do not match. Got {} and {}.".format(depth_maps.shape, optimised_depth_maps.shape)

    assert video_data.frames.shape[0] == depth_maps.shape[0], \
        "Number of video frames and depth maps are not equal. " \
        "Got {} and {}.".format(video_data.shape[0], depth_maps.shape[0])

    if video_data.frames.shape[-2:] != depth_maps.shape[-2:]:
        warnings.warn(
            "Video spatial dimensions did not match the spatial dimensions of the depth maps. "
            "Got {video} shaped video frames and {depth} shaped depth maps. "
            "Resizing the video frames to {depth}.".format(video=video_data.frames.shape[-2:],
                                                           depth=depth_maps.shape[-2:])
        )

        height, width = depth_maps.shape[-2:]

        frames = F.interpolate(torch.from_numpy(video_data.frames),
                               size=(height, width), mode="bilinear", align_corners=True)
        frames = frames.numpy()
    else:
        frames = video_data.frames

    stacked_frame_shape = list(frames.shape)
    stacked_frame_shape[-1] *= 3
    stacked_frame_shape = tuple(stacked_frame_shape)
    logger.log("Stacked video frames and depth maps. Stacked video shape (NCHW): {}.".format(stacked_frame_shape))

    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    video_writer = cv2.VideoWriter(sample_video_path, fourcc, video_data.fps,
                                   (stacked_frame_shape[3], stacked_frame_shape[2]))

    def gen_batches(batch_size=128, queue_size=1):
        num_batches = math.ceil(len(frames) / batch_size)

        def get_batch(batch_i):
            batch_start = batch_i * batch_size

            if batch_i < num_batches - 1:
                batch_end = batch_start + batch_size
            else:
                batch_end = batch_start + len(frames) % batch_size

            frame = frames[batch_start:batch_end]
            depth_map = depth_maps_transform(depth_maps[batch_start:batch_end])
            optimised_depth_map = optimised_depth_maps_transform(optimised_depth_maps[batch_start:batch_end])

            stacked_frames = np.concatenate(
                (frame, depth_map, optimised_depth_map),
                axis=-1
            )

            # Transpose to NHWC for OpenCV.
            stacked_frames = stacked_frames.transpose((0, 2, 3, 1))

            return stacked_frames

        with ThreadPoolExecutor() as executor:
            batch_queue = deque([executor.submit(get_batch, i) for i in range(queue_size)])

            for next_batch_i in range(queue_size, num_batches):
                batch_queue.append(executor.submit(get_batch, next_batch_i))
                yield batch_queue.popleft().result()

            while len(batch_queue) > 0:
                yield batch_queue.popleft().result()

    num_frames_written = 0

    for batch in gen_batches(batch_size=128, queue_size=2):
        for stacked_frame in batch:
            video_writer.write(stacked_frame)

            num_frames_written += 1
            logger.log("Wrote {}/{} frames.\r".format(num_frames_written, len(frames)), end="")

    print()

    video_writer.release()
    logger.log("Wrote video to {}.".format(video_path))


if __name__ == '__main__':
    plac.call(main)
