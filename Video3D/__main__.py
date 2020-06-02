import math
import os
import pickle
from typing import List

import cv2
import numpy as np
import plac
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

import sys
sys.path.insert(0, '..')

from MiDaS.models.midas_net import MidasNet
from MiDaS.models.transforms import Resize, NormalizeImage, PrepareForNet
from Video3D.align_images import align_images
from Video3D.colmap_parsing import parse_cameras, parse_images, parse_points
from Video3D.dataset import OpticalFlowDatasetBuilder
from models import FlowNet2
from utils.tools import TimerBlock


def sample_frame_pairs(num_frames):
    frame_pairs = {(i, i + 1) for i in range(num_frames - 1)}

    for l in range(math.floor(math.log2(num_frames - 1)) + 1):
        sl = {(i, i + 2 ** l) for i in range(num_frames - 2 ** l) if i % 2 ** (l - 1) == 0}
        frame_pairs = frame_pairs.union(sl)

    return frame_pairs


class NumpyDataset(Dataset):
    def __init__(self, arrays: List[np.ndarray], to_tensor_transform: Compose):
        """Dataset wrapping numpy arrays.

        Each sample will be retrieved by indexing tensors along the first dimension.

        :param arrays: numpy arrays that all have the same shape.
        :param to_tensor_transform: The transform that takes a numpy array and converts it to a torch Tensor object.
        """
        assert all(arrays[0].shape == tensor.shape for tensor in arrays)
        self.arrays = arrays
        self.to_tensor = to_tensor_transform

    def __getitem__(self, index):
        return self.to_tensor({"image": self.arrays[index]})

    def __len__(self):
        return len(self.arrays)


# TODO: Add flag for clearing cache.
def generate_depth_maps(depth_estimation_model_path, frames, width, height, logger, dnn_depth_map_cache_path):
    try:
        depth_maps = np.load(dnn_depth_map_cache_path)
        logger.log("Loaded DNN depth maps from {}.".format(dnn_depth_map_cache_path))
    except IOError:
        transform = Compose(
            [
                Resize(
                    width,
                    height,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        logger.log("Created image transform.")

        video_dataset = NumpyDataset(frames, transform)
        logger.log("Created dataset.")

        model = MidasNet(depth_estimation_model_path, non_negative=True)
        model = model.cuda(0)
        model.load_state_dict(torch.load(depth_estimation_model_path, map_location="cpu"))
        model.eval()
        logger.log("Loaded depth estimation model.")

        depth_maps = []
        sequential_dataloader = DataLoader(video_dataset, batch_size=8, shuffle=False)
        frame_i = 0

        with torch.no_grad():
            for batch in sequential_dataloader:
                imgs = batch["image"]
                imgs = imgs.cuda()
                prediction = model(imgs)
                depth_maps.append(prediction.cpu())

                frame_i += imgs.shape[0]
                logger.log("Generated {:02d} frames.\r".format(frame_i), end="")

        logger.log("Generated {} depth maps.".format(len(depth_maps)))

        depth_maps = torch.cat(depth_maps, dim=0)
        logger.log("Concatenated depth maps.")

        output_shape = depth_maps.shape[-2:]

        depth_maps = torch.nn.functional.interpolate(depth_maps.unsqueeze(1), size=(width, height),
                                                     mode='bilinear', align_corners=True)

        # Network produces output as NCWH - dropped channel since output is B&W and just need to change to NHW.
        depth_maps = depth_maps.squeeze().permute((0, 2, 1)).to(torch.float32).cpu().numpy()
        logger.log("Resized depth maps back to {} from {}.".format(depth_maps.shape[-2:], output_shape))

        np.save(dnn_depth_map_cache_path, depth_maps)
        logger.log("Saved DNN depth maps to {}.".format(dnn_depth_map_cache_path))

    return depth_maps


def generate_sparse_depth_maps(camera, points2d_by_image, points3d_by_id, logger, sparse_depth_map_cache_path):
    try:
        sparse_depth_maps = np.load(sparse_depth_map_cache_path)
        logger.log("Loaded sparse depth maps from {}.".format(sparse_depth_map_cache_path))
    except IOError:
        sparse_depth_maps = []

        for points in points2d_by_image.values():
            depth_map = np.zeros(shape=camera.shape, dtype=np.float32)

            for point in points:
                if point.point3d_id > -1 and point.x <= depth_map.shape[1] and point.y <= depth_map.shape[0]:
                    point3d = points3d_by_id[point.point3d_id]

                    depth_map[int(point.y), int(point.x)] = point3d.z

            sparse_depth_maps.append(depth_map)

        sparse_depth_maps = np.array(sparse_depth_maps, dtype=np.float32)
        logger.log("Generated sparse depth maps for each frame.")

        np.save(sparse_depth_map_cache_path, sparse_depth_maps)
        logger.log("Saved sparse depth maps to {}.".format(sparse_depth_map_cache_path))
    return sparse_depth_maps


class CustomResize(object):
    """Resize sample to given size (width, height)."""

    def __init__(
            self,
            width,
            height,
            resize_target=True,
            keep_aspect_ratio=False,
            ensure_multiple_of=1,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    "resize_method {} not implemented".format(self.__resize_method)
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError("resize_method {} not implemented".format(self.__resize_method))

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample.shape[1], sample.shape[0]
        )

        # resize sample
        sample = cv2.resize(
            sample,
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        return sample


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

    with TimerBlock("Parse COLMAP Output") as block:
        try:
            with open(colmap_cache_path, 'rb') as f:
                camera, poses_by_image, points2d_by_image, points3d_by_id = pickle.load(f)

            block.log("Loaded parsed COLMAP output from {}.".format(colmap_cache_path))
        except FileNotFoundError:
            cameras_txt = os.path.join(colmap_output_path, "cameras.txt")
            images_txt = os.path.join(colmap_output_path, "images.txt")
            points_3d_txt = os.path.join(colmap_output_path, "points3D.txt")

            camera = parse_cameras(cameras_txt)[0]
            block.log("Parsed camera intrinsics.")

            poses_by_image, points2d_by_image = parse_images(images_txt)
            block.log("Parsed camera poses and 2D points.")

            points3d_by_id = parse_points(points_3d_txt)
            block.log("Parsed 3D points.")

            if not os.path.isdir(cache_path):
                os.makedirs(cache_path)

            with open(colmap_cache_path, 'wb') as f:
                pickle.dump((camera, poses_by_image, points2d_by_image, points3d_by_id), f)

            block.log("Saved parsed COLMAP output to {}.".format(colmap_cache_path))

    with TimerBlock("Load Video") as block:
        input_video = cv2.VideoCapture(video_path)

        if not input_video.isOpened():
            raise RuntimeError("Could not open video from the path {}.".format(video_path))

        block.log("Opened video from path the {}.".format(video_path))

        frames = []

        while input_video.isOpened():
            has_frame, frame = input_video.read()

            if not has_frame:
                break

            frames.append(frame)

        block.log("Extracted video frames.")

        width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        block.log("Got video dimensions: {}x{}.".format(width, height))

        num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        block.log("Got number of frames: {}.".format(num_frames))

        if input_video.isOpened():
            input_video.release()

    with TimerBlock("Calculate Relative Depth Scaling Factor") as block:
        try:
            with open(relative_depth_scale_cache_path, 'rb') as f:
                relative_depth_scale = pickle.load(f)

            block.log("Loaded relative depth scale from {}".format(relative_depth_scale_cache_path))
        except FileNotFoundError:
            depth_maps = generate_depth_maps(depth_estimation_model_path, frames, width, height, block,
                                             dnn_depth_map_cache_path)

            sparse_depth_maps = generate_sparse_depth_maps(camera, points2d_by_image, points3d_by_id, block,
                                                           sparse_depth_map_cache_path)

            zero_mask = sparse_depth_maps == 0

            relative_depth_scale = np.divide(depth_maps, sparse_depth_maps, dtype=np.float32)
            relative_depth_scale[zero_mask] = np.nan
            relative_depth_scale = np.nanmedian(relative_depth_scale, axis=[1, 2])
            relative_depth_scale = np.mean(relative_depth_scale)
            block.log("Calculated relative depth scale factor: {:.2f}.".format(relative_depth_scale))

            del depth_maps
            del sparse_depth_maps
            block.log("Freed memory used by depth maps.")

            with open(relative_depth_scale_cache_path, 'wb') as f:
                pickle.dump(relative_depth_scale, f)

            block.log("Saved relative depth scale factor to {}.".format(relative_depth_scale_cache_path))

        for pose in poses_by_image.values():
            pose.t = relative_depth_scale * pose.t

        block.log("Scaled the translation component of the camera poses by the relative scale factor of {:.2f}"
                  .format(relative_depth_scale))

    with TimerBlock("Create Optical Flow Dataset") as block:
        frame_pair_indexes = sample_frame_pairs(num_frames)
        block.log("Sampled frame pairs")

        # TODO: Make this less hacky...
        # Simulate argparse object because I'm too lazy to implement it properly.
        class Object(object):
            pass

        args = Object()
        args.fp16 = False
        args.rgb_max = 255

        t = CustomResize(
            width,
            height,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=64,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )

        flow_net = FlowNet2(args).cuda()
        flow_net.load_state_dict(torch.load(optical_flow_model_path)["state_dict"])
        block.log("Loaded optical flow model from {}.".format(optical_flow_model_path))

        if not os.path.isdir(dataset_cache_path):
            os.makedirs(dataset_cache_path)

            num_filtered = 0

            def calculate_optical_flow(frame_i, frame_j):
                images = list(map(t, (frame_i, frame_j)))
                images = np.array(images).transpose(3, 0, 1, 2)
                images_tensor = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

                optical_flow = flow_net(images_tensor)
                optical_flow = torch.nn.functional.interpolate(optical_flow, size=(height, width),
                                                               mode='bilinear', align_corners=True)
                optical_flow = optical_flow.squeeze().permute((1, 2, 0)).cpu().numpy()

                return optical_flow

            with torch.no_grad(), OpticalFlowDatasetBuilder(dataset_cache_path, camera) as dataset_builder:
                for pair_index, (i, j) in enumerate(frame_pair_indexes):
                    frame_i = frames[i]
                    frame_j = frames[j]
                    frame_j_aligned = align_images(frame_i, frame_j)

                    optical_flow_forwards = calculate_optical_flow(frame_i, frame_j_aligned)

                    optical_flow_backwards = calculate_optical_flow(align_images(frames[j], frames[i]), frames[j])

                    delta = np.abs(optical_flow_forwards - optical_flow_backwards)
                    valid_mask = (delta <= 1).astype(np.bool)
                    # `valid_mask` is up to this point, indicating if each of the u and v components of each optical
                    # flow vector are within 1px error.
                    # However, we need it to indicate this on a per-pixel basis, so we combine the binary maps of the u
                    # and v components to give us the validity of the optical flow at the given pixel.
                    valid_mask = valid_mask[:, :, 0] & valid_mask[:, :, 1]

                    # TODO: Check if `delta = np.sum(np.abs(a - b), axis=-1) <= 1` would do the same thing as above.
                    should_keep_frame = np.mean(valid_mask) >= 0.8

                    if should_keep_frame:
                        dataset_builder.add(i, j, frame_i, frame_j, optical_flow_forwards, valid_mask, poses_by_image)
                    else:
                        num_filtered += 1

                    block.log("Processed {} frame pairs out of {} ({} kept, {} filtered out).\r"
                              .format(pair_index + 1, len(frame_pair_indexes), pair_index + 1 - num_filtered,
                                      num_filtered), end="")

            print()
            block.log("Saved dataset to {}.".format(dataset_cache_path))
        else:
            block.log("Found dataset at {}.".format(dataset_cache_path))

    with TimerBlock("Fine Tune Depth Estimation Model") as block:
        # TODO: Load dataset
        # TODO: Load depth estimation network
        # TODO: Define loss functions
        # TODO: Optimise depth estimation network
        # TODO: Save optimised depth estimation network weights
        pass

    with TimerBlock("Generate Refined Depth Maps") as block:
        # TODO: Generate depth maps for each frame.
        pass


if __name__ == '__main__':
    plac.call(main)
