import json
import pickle
from collections import defaultdict
from typing import List, Tuple, DefaultDict, Optional

import numpy as np
from scipy.spatial.transform import Rotation


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
            [self.focal_length, self.skew, self.center_x],
            [0.0, self.focal_length, self.center_y],
            [0.0, 0.0, 1.0]
        ])

    @staticmethod
    def to_homogeneous_matrix(m):
        assert len(m.shape) == 2 and m.shape[0] == m.shape[1], \
            "The matrix `m` must be a square matrix, got dimensions {}.".format(m.shape)

        m_ = np.zeros((m.shape[0], m.shape[0] + 1))
        m_[:, :-1] = m

        return m_

    def get_inverse_matrix(self):
        return np.linalg.inv(self.get_matrix())

    def save_json(self, f):
        if isinstance(f, str):
            with open(f, 'w') as fp:
                json.dump(self.__dict__, fp)
        else:
            json.dump(self.__dict__, f)

    @staticmethod
    def load_json(f):
        if isinstance(f, str):
            with open(f, 'r') as fp:
                args = json.load(fp)
        else:
            args = json.load(f)

        return Camera(**args)

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

    def save_pkl(self, f):
        if isinstance(f, str):
            with open(f, 'wb') as fp:
                pickle.dump(self, fp)
        else:
            pickle.dump(self, f)

    @staticmethod
    def load_pkl(f):
        if isinstance(f, str):
            with open(f, 'rb') as fp:
                return pickle.load(fp)
        else:
            return pickle.load(f)


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
    poses = defaultdict(CameraPose)
    points = defaultdict(list)

    with open(txt_file, 'r') as f:
        while True:
            line1 = f.readline()

            if line1 and line1[0] == "#":
                continue

            line2 = f.readline()

            if not line1 or not line2:
                break

            # COLMAP image IDs start at one, everything I do is zero-indexed so need to subtract one here.
            image_id = int(line1.split()[0]) - 1

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
            # COLMAP image IDs start at one, everything I do is zero-indexed so need to subtract one here.
            image_id = int(parts[i]) - 1
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
