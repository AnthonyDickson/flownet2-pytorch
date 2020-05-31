import math
from collections import defaultdict
from typing import List, Tuple, DefaultDict, Optional

import cv2
import plac
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torchvision.transforms import Compose

from MiDaS.models.midas_net import MidasNet
from MiDaS.models.transforms import Resize, NormalizeImage, PrepareForNet
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


@plac.annotations(
    cameras_txt=plac.Annotation("The camera intrinsics txt file exported from COLMAP."),
    images_txt=plac.Annotation("The image data txt file exported from COLMAP."),
    points_3d_txt=plac.Annotation("The 3D points txt file exported from COLMAP."),
    video_path=plac.Annotation('The path to the source video file.', type=str,
                               kind='option', abbrev='i'),
    model_path=plac.Annotation("The path to the pretrained MiDaS model weights.", type=str, kind="option", abbrev='m'),
)
def main(cameras_txt: str, images_txt: str, points_3d_txt: str, video_path: str, model_path: str = "model.pt"):
    camera = parse_cameras(cameras_txt)[0]
    poses_by_image, points2d_by_image = parse_images(images_txt)
    points3d_by_id = parse_points(points_3d_txt)

    depth_maps = []

    for points in points2d_by_image.values():
        depth_map = np.zeros(shape=camera.shape)

        for point in points:
            if point.point3d_id > -1 and point.x <= depth_map.shape[1] and point.y <= depth_map.shape[0]:
                point3d = points3d_by_id[point.point3d_id]

                depth_map[int(point.y), int(point.x)] = point3d.z

        depth_maps.append(depth_map)

    relative_depth_scales = []

    with TimerBlock("Load Video and Depth Estimation Model") as block:
        input_video = cv2.VideoCapture(video_path)
        frame_i = 0

        if not input_video.isOpened():
            raise RuntimeError(f"Could not open video from path `{video_path}`.")

        block.log(f"Opened video from path `{video_path}`.")

        width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        block.log(f"Got video dimensions: {width}x{height}.")

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

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = MidasNet(model_path, non_negative=True)
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        block.log("Loaded depth estimation model.")

    with TimerBlock("Calculate Relative Depth Scaling Factor") as block:
        while input_video.isOpened():
            has_frame, frame = input_video.read()

            if not has_frame:
                break

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            img_input = transform({"image": frame})["image"]

            with torch.no_grad():
                sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
                prediction = model.forward(sample)

                output = torch.nn.functional.interpolate(prediction.unsqueeze(0), size=(width, height),
                                                         mode='bilinear', align_corners=True)
                output = output.squeeze().permute((1, 0))
                output = output.cpu().numpy()

                relative_scales = []
                sparse_depth = depth_maps[frame_i]

                for row in range(height):
                    for col in range(width):
                        if sparse_depth[row, col] != 0:
                            relative_scales.append(output[row, col] / sparse_depth[row, col])

                relative_depth_scales.append(np.median(relative_scales))

            block.log(f"Frame {frame_i:02d}\r", end="")
            frame_i += 1

        print()
        block.log(f"Processed {frame_i} frames.")

        relative_depth_scale = np.mean(relative_depth_scales)

        block.log(f"Calculated relative depth scale factor: {relative_depth_scale:.2f}.")

    with TimerBlock("Sample Frame Pairs") as block:
        num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_pairs = sample_frame_pairs(num_frames)
        block.log("Sampled frame pairs")

    with TimerBlock("Align Frame Pairs") as block:
        pass

    with TimerBlock("Generate Dense Optical Flow") as block:
        pass

    with TimerBlock("Filter Frame Pairs") as block:
        pass

    with TimerBlock("Fine Tune Depth Estimation Model") as block:
        pass

    with TimerBlock("Generate Depth Maps") as block:
        pass

    if input_video.isOpened():
        input_video.release()


if __name__ == '__main__':
    plac.call(main)
