import math
import os
import pickle
from collections import defaultdict
from typing import List, Tuple, DefaultDict, Optional, Iterable, Collection, Container, Union

import cv2
import plac
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import Compose

from MiDaS.models.midas_net import MidasNet
from MiDaS.models.transforms import Resize, NormalizeImage, PrepareForNet
from align_images import align_images
from utils.tools import TimerBlock


class Camera:
    def __init__(self, width, height, focal_length, center_x, center_y, skew):
        self.width = int(width)
        self.height = int(height)
        self.focal_length = float(focal_length)
        self.center_x = int(center_x)
        self.center_y = int(center_y)
        self.skew = float(skew)

    @property
    def shape(self):
        return self.height, self.width

    def get_matrix(self):
        return np.array([
            [self.focal_length, self.skew, self.center_x, 0.0],
            [0.0, self.focal_length, self.center_y, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])

    @staticmethod
    def parse_line(line: str) -> 'Camera':
        parts = line.split()
        parts = map(float, parts[2:])
        width, height, focal_length, center_x, center_y, skew = list(parts)

        return Camera(width, height, focal_length, center_x, center_y, skew)


def parse_cameras(txt_file):
    cameras = []

    with open(txt_file, 'r') as f:
        for line in f:
            if line[0] == "#":
                continue

            camera = Camera.parse_line(line)
            cameras.append(camera)

    return cameras


class CameraPose:
    def __init__(self, R=Rotation.identity(), t=np.zeros(shape=(3, 1))):
        self.R = R
        self.t = t

    @staticmethod
    def parse_COLMAP_txt(line: str):
        line_parts = line.split()
        line_parts = map(float, line_parts[1:-2])
        qw, qx, qy, qz, tx, ty, tz = tuple(line_parts)

        R = Rotation.from_quat([qx, qy, qz, qw])
        t = np.array([[tx, ty, tz]]).T

        return CameraPose(R, t)


class Point2D:
    def __init__(self, x: float, y: float, point3d_id: int):
        self.x = x
        self.y = y
        self.point3d_id = point3d_id

    @staticmethod
    def parse_line(line: str) -> List['Point2D']:
        parts = line.split()
        points = []

        for i in range(0, len(parts), 3):
            x = float(parts[i])
            y = float(parts[i + 1])
            point3d_id = int(parts[i + 2])

            points.append(Point2D(x, y, point3d_id))

        return points


def parse_images(txt_file) -> Tuple[DefaultDict[int, Optional[CameraPose]], DefaultDict[int, List[Point2D]]]:
    poses = defaultdict(Camera)
    points = defaultdict(list)

    with open(txt_file, 'r') as f:
        while True:
            line1 = f.readline()

            if line1 and line1[0] == "#":
                continue

            line2 = f.readline()

            if not line1 or not line2:
                break

            image_id = int(line1.split()[0])

            pose = CameraPose.parse_COLMAP_txt(line1)
            points_in_image = Point2D.parse_line(line2)

            poses[image_id] = pose
            points[image_id] = points_in_image

    return poses, points


class Track:
    def __init__(self, image_id: int, point2d_index: int):
        self.image_id = int(image_id)
        self.point2d_index = int(point2d_index)

    @staticmethod
    def parse_line(line: str) -> List['Track']:
        parts = line.split()

        return Track.parse_strings(parts)

    @staticmethod
    def parse_strings(parts: List[str]) -> List['Track']:
        tracks = []

        for i in range(0, len(parts), 2):
            image_id = int(parts[i])
            point2d_index = int(parts[i + 1])

            tracks.append(Track(image_id, point2d_index))

        return tracks


class Point3D:
    def __init__(self, point3d_id: int = -1,
                 x: float = 0.0, y: float = 0.0, z: float = 0.0,
                 r: int = 0, g: int = 0, b: int = 0,
                 error: float = float('nan'), track=None):
        if track is None:
            track = []

        self.point3d_id = int(point3d_id)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.error = error
        self.track = track

    @staticmethod
    def parse_line(line: str) -> 'Point3D':
        parts = line.split()

        point3d_id, x, y, z, r, g, b, error = parts[:8]
        track = parts[8:]

        return Point3D(point3d_id, x, y, z, r, g, b, error, Track.parse_strings(track))


def parse_points(txt_file) -> DefaultDict[int, Optional[Point3D]]:
    points = defaultdict(Point3D)

    with open(txt_file, 'r') as f:
        for line in f:
            if line[0] == "#":
                continue

            point = Point3D.parse_line(line)
            points[point.point3d_id] = point

    return points


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
def generate_depth_maps(model, video_dataset, width, height, logger, dnn_depth_map_cache_path):
    try:
        depth_maps = np.load(dnn_depth_map_cache_path)
        logger.log(f"Loaded DNN depth maps from {dnn_depth_map_cache_path}.")
    except IOError:
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
                logger.log(f"Generated {frame_i:02d} frames.\r", end="")

        logger.log(f"Generated {len(depth_maps)} depth maps.")

        depth_maps = torch.cat(depth_maps, dim=0)
        logger.log("Concatenated depth maps.")

        output_shape = depth_maps.shape[-2:]

        depth_maps = torch.nn.functional.interpolate(depth_maps.unsqueeze(1), size=(width, height),
                                                     mode='bilinear', align_corners=True)

        # Network produces output as NCWH - dropped channel since output is B&W and just need to change to NHW.
        depth_maps = depth_maps.squeeze().permute((0, 2, 1)).to(torch.float32).cpu().numpy()
        logger.log(f"Resized depth maps back to {depth_maps.shape[-2:]} from {output_shape}.")

        np.save(dnn_depth_map_cache_path, depth_maps)
        logger.log(f"Saved DNN depth maps to {dnn_depth_map_cache_path}.")
    return depth_maps


def generate_sparse_depth_maps(camera, points2d_by_image, points3d_by_id, logger, sparse_depth_map_cache_path):
    try:
        sparse_depth_maps = np.load(sparse_depth_map_cache_path)
        logger.log(f"Loaded sparse depth maps from {sparse_depth_map_cache_path}.")
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
        logger.log(f"Saved sparse depth maps to {sparse_depth_map_cache_path}.")
    return sparse_depth_maps


@plac.annotations(
    colmap_output_path=plac.Annotation(
        "The path to the folder containing the text files `cameras.txt`, `images.txt` and `points3D.txt`."),
    video_path=plac.Annotation('The path to the source video file.', type=str,
                               kind='option', abbrev='i'),
    model_path=plac.Annotation("The path to the pretrained MiDaS model weights.", type=str, kind="option", abbrev='m'),
    cache_path=plac.Annotation("Where to save and load intermediate results to and from.", type=str, kind="option",
                               abbrev="c")
)
def main(colmap_output_path: str, video_path: str, model_path: str = "model.pt", cache_path: str = ".cache"):
    cache_path = os.path.abspath(cache_path)
    video_name = os.path.basename(video_path)
    cache_path = os.path.join(cache_path, video_name)
    colmap_cache_path = os.path.join(cache_path, "parsed_colmap_output.pkl")
    sparse_depth_map_cache_path = os.path.join(cache_path, "sparse_depth_map.npy")
    dnn_depth_map_cache_path = os.path.join(cache_path, "dnn_depth_map.npy")
    relative_depth_scale_cache_path = os.path.join(cache_path, "relative_depth_scale.pkl")
    # aligned_frame_pairs_cache_path = os.path.join(cache_path, "aligned_frame_pairs.npy")

    with TimerBlock("Parse COLMAP Output") as block:
        try:
            with open(colmap_cache_path, 'rb') as f:
                camera, poses_by_image, points2d_by_image, points3d_by_id = pickle.load(f)

            block.log(f"Loaded parsed COLMAP output from {colmap_cache_path}.")
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

            block.log(f"Saved parsed COLMAP output to {colmap_cache_path}.")

    with TimerBlock("Load Video") as block:
        input_video = cv2.VideoCapture(video_path)

        if not input_video.isOpened():
            raise RuntimeError(f"Could not open video from the path {video_path}.")

        block.log(f"Opened video from path the {video_path}.")

        frames = []

        while input_video.isOpened():
            has_frame, frame = input_video.read()

            if not has_frame:
                break

            frames.append(frame)

        block.log("Extracted video frames.")

        width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        block.log(f"Got video dimensions: {width}x{height}.")

        num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        block.log(f"Got number of frames: {num_frames}.")

        if input_video.isOpened():
            input_video.release()

    with TimerBlock("Calculate Relative Depth Scaling Factor") as block:
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

        block.log("Created image transform.")

        video_dataset = NumpyDataset(frames, transform)
        block.log("Created dataset.")

        model = MidasNet(model_path, non_negative=True)
        model = model.cuda(0)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        block.log("Loaded depth estimation model.")

        try:
            with open(relative_depth_scale_cache_path, 'rb') as f:
                relative_depth_scale = pickle.load(f)

            block.log(f"Loaded relative depth scale from {relative_depth_scale_cache_path}")
        except FileNotFoundError:
            depth_maps = generate_depth_maps(model, video_dataset, width, height, block, dnn_depth_map_cache_path)

            sparse_depth_maps = generate_sparse_depth_maps(camera, points2d_by_image, points3d_by_id, block,
                                                           sparse_depth_map_cache_path)

            zero_mask = sparse_depth_maps == 0

            relative_depth_scale = np.divide(depth_maps, sparse_depth_maps, dtype=np.float32)
            relative_depth_scale[zero_mask] = np.nan
            relative_depth_scale = np.nanmedian(relative_depth_scale, axis=[1, 2])
            relative_depth_scale = np.mean(relative_depth_scale)
            block.log(f"Calculated relative depth scale factor: {relative_depth_scale:.2f}.")

            with open(relative_depth_scale_cache_path, 'wb') as f:
                pickle.dump(relative_depth_scale, f)

            block.log(f"Saved relative depth scale factor to {relative_depth_scale_cache_path}.")

    with TimerBlock("Generate Dense Optical Flow") as block:
        frame_pair_indexes = sample_frame_pairs(num_frames)
        block.log("Sampled frame pairs")

        # try:
        #     aligned_frame_pairs = np.load(aligned_frame_pairs_cache_path)
        #     block.log(f"Loaded aligned frame pairs from {aligned_frame_pairs_cache_path}.")
        # except IOError:
        #     aligned_frame_pairs = []
        #
        #     for pair_index, (i, j) in enumerate(frame_pair_indexes):
        #         aligned_frame = align_images(frames[i], frames[j])
        #         aligned_frame_pairs.append((pair_index, aligned_frame.astype(np.float32)))
        #         block.log(f"Aligned {pair_index + 1} out of {len(frame_pair_indexes)} frame pairs.\r", end="")
        #
        #     block.log(f"Aligned {len(aligned_frame_pairs)} frame pairs.")
        #
        #     # # TODO: Fix out-of-memory issues with concatenating aligned frame pairs
        #     # aligned_frame_pairs = np.array(aligned_frame_pairs, dtype=np.float32)
        #     # np.save(aligned_frame_pairs_cache_path, aligned_frame_pairs)
        #     with open(aligned_frame_pairs_cache_path, 'wb') as f:
        #         pickle.dump(aligned_frame_pairs)
        #
        #     block.log(f"Saved aligned frame pairs to {aligned_frame_pairs_cache_path}.")

    with TimerBlock("Fine Tune Depth Estimation Model") as block:
        pass

    with TimerBlock("Generate Refined Depth Maps") as block:
        pass


if __name__ == '__main__':
    plac.call(main)
