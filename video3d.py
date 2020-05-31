import math
import os
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
    def __init__(self, R, t):
        self.R = R
        self.t = t

    @staticmethod
    def parse_COLMAP_txt(line: str):
        line_parts = line.split()
        line_parts = map(float, line_parts[1:-2])
        qw, qx, qy, qz, tx, ty, tz = tuple(line_parts)

        R = Rotation.from_quat([qx, qy, qz, qw])
        t = np.array([tx, ty, tz]).T

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
    poses = defaultdict(lambda: None)
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
    def __init__(self, point3d_id: int, x: float, y: float, z: float, r: int, g: int, b: int, error: float,
                 track: List[Track]):
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
    points = defaultdict(lambda: None)

    with open(txt_file, 'r') as f:
        for line in f:
            if line[0] == "#":
                continue

            point = Point3D.parse_line(line)
            points[point.point3d_id] = point

    return points


def sample_frame_pairs(num_frames):
    frame_pairs = {(i, i + 1) for i in range(num_frames)}

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


@plac.annotations(
    # cameras_txt=plac.Annotation("The camera intrinsics txt file exported from COLMAP."),
    # images_txt=plac.Annotation("The image data txt file exported from COLMAP."),
    # points_3d_txt=plac.Annotation("The 3D points txt file exported from COLMAP."),
    colmap_output_path=plac.Annotation(
        "The path to the folder containing the text files `cameras.txt`, `images.txt` and `points3D.txt`."),
    video_path=plac.Annotation('The path to the source video file.', type=str,
                               kind='option', abbrev='i'),
    model_path=plac.Annotation("The path to the pretrained MiDaS model weights.", type=str, kind="option", abbrev='m'),
)
def main(colmap_output_path: str, video_path: str, model_path: str = "model.pt"):
    with TimerBlock("Parse COLMAP Output") as block:
        cameras_txt = os.path.join(colmap_output_path, "cameras.txt")
        images_txt = os.path.join(colmap_output_path, "images.txt")
        points_3d_txt = os.path.join(colmap_output_path, "points3D.txt")

        camera = parse_cameras(cameras_txt)[0]
        block.log("Parsed camera intrinsics.")

        poses_by_image, points2d_by_image = parse_images(images_txt)
        block.log("Parsed camera poses and 2D points.")

        points3d_by_id = parse_points(points_3d_txt)
        block.log("Parsed 3D points.")

        sparse_depth_maps = []

        for points in points2d_by_image.values():
            depth_map = np.zeros(shape=camera.shape)

            for point in points:
                if point.point3d_id > -1 and point.x <= depth_map.shape[1] and point.y <= depth_map.shape[0]:
                    point3d = points3d_by_id[point.point3d_id]

                    depth_map[int(point.y), int(point.x)] = point3d.z

            sparse_depth_maps.append(depth_map)

        sparse_depth_maps = np.array(sparse_depth_maps)

        block.log("Generated sparse depth maps for each frame.")

    with TimerBlock("Load Video and Depth Estimation Model") as block:
        input_video = cv2.VideoCapture(video_path)
        frame_i = 0

        if not input_video.isOpened():
            raise RuntimeError(f"Could not open video from path `{video_path}`.")

        block.log(f"Opened video from path `{video_path}`.")

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

    with TimerBlock("Calculate Relative Depth Scaling Factor") as block:
        relative_depth_scales = []
        depth_maps = []
        sequential_dataloader = DataLoader(video_dataset, batch_size=8, shuffle=False)

        with torch.no_grad():
            for batch in sequential_dataloader:
                imgs = batch["image"]
                imgs = imgs.cuda()
                prediction = model(imgs)
                depth_maps.append(prediction.cpu())

                frame_i += imgs.shape[0]
                block.log(f"Frame {frame_i:02d}\r", end="")

        print()
        block.log(f"Generated {len(depth_maps)} depth maps.")

        stacked_depth_maps = torch.stack(depth_maps, dim=0)

        depth_maps = torch.nn.functional.interpolate(stacked_depth_maps, size=(width, height),
                                                     mode='bilinear', align_corners=True)
        depth_maps = depth_maps.squeeze().permute((1, 0))
        depth_maps = depth_maps.cpu().numpy()

        relative_depth_scale = np.mean(relative_depth_scales)

        block.log(f"Calculated relative depth scale factor: {relative_depth_scale:.2f}.")

    with TimerBlock("Sample Frame Pairs") as block:
        frame_pair_indexes = sample_frame_pairs(num_frames)
        block.log("Sampled frame pairs")

    with TimerBlock("Align Frame Pairs") as block:
        frame_pairs = []

        for i, j in frame_pair_indexes:
            frame_pairs.append((frames[i], align_images(frames[i], frames[j])))

        block.log(f"Aligned {len(frame_pairs)} frame pairs.")

    with TimerBlock("Generate Dense Optical Flow") as block:
        pass

    with TimerBlock("Filter Frame Pairs") as block:
        pass

    with TimerBlock("Fine Tune Depth Estimation Model") as block:
        pass

    with TimerBlock("Generate Depth Maps") as block:
        pass


if __name__ == '__main__':
    plac.call(main)
