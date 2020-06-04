import json
import os
import pickle
import warnings
from typing import Optional, List

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from MiDaS.models.transforms import Resize, NormalizeImage, PrepareForNet
from Video3D.colmap_parsing import Camera, CameraPose


class OpticalFlowDataMetadata:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.frames_metadata_path = os.path.join(self.base_dir, "frames.json")
        self.poses_metadata_path = os.path.join(self.base_dir, "poses.json")
        self.camera_intrinsics_metadata_path = os.path.join(self.base_dir, "camera_intrinsics.json")
        self.optical_flow_fields_metadata_path = os.path.join(self.base_dir, "optical_flow_fields.json")

        self.frame_paths = dict()
        self.pose_paths = dict()
        self.optical_flow_paths = dict()
        self.camera = None  # type: Optional[Camera]
        self.frame_indexes = set()
        self.frame_pair_indexes = list()
        self.length = -1

        # TODO: Add frame size and other info (e.g. num. frames, num. frame pairs) to metadata.

    def make_dirs(self):
        os.makedirs(self.base_dir)

    def set_camera(self, camera):
        self.camera = camera

    def add_frame(self, frame_index, frame_path, pose_path):
        self.frame_paths[frame_index] = frame_path
        self.pose_paths[frame_index] = pose_path

    def add_optical_flow(self, i, j, optical_flow_path):
        if i not in self.optical_flow_paths:
            self.optical_flow_paths[i] = dict()

        self.optical_flow_paths[i][j] = optical_flow_path

    def save(self):
        with open(self.frames_metadata_path, 'w') as f:
            json.dump(self.frame_paths, f)

        with open(self.poses_metadata_path, 'w') as f:
            json.dump(self.pose_paths, f)

        with open(self.optical_flow_fields_metadata_path, 'w') as f:
            json.dump(self.optical_flow_paths, f)

        self.camera.save_json(self.camera_intrinsics_metadata_path)

    def _apply_key(self, fn, d, recurse=False):
        if isinstance(d, dict):
            if recurse:
                return {fn(key): self._apply_key(fn, value) for key, value in d.items()}
            else:
                return {fn(key): value for key, value in d.items()}
        else:
            return d

    def load(self):
        with open(self.frames_metadata_path, 'r') as f:
            self.frame_paths = json.load(f)
            self.frame_paths = self._apply_key(int, self.frame_paths)

        with open(self.poses_metadata_path, 'r') as f:
            self.pose_paths = json.load(f)
            self.pose_paths = self._apply_key(int, self.pose_paths)

        with open(self.optical_flow_fields_metadata_path, 'r') as f:
            self.optical_flow_paths = json.load(f)
            self.optical_flow_paths = self._apply_key(int, self.optical_flow_paths, recurse=True)

        self.camera = Camera.load_json(self.camera_intrinsics_metadata_path)

        self.frame_indexes = set(map(int, self.frame_paths.keys()))
        self.frame_pair_indexes = []

        for i in self.optical_flow_paths:
            for j in self.optical_flow_paths[i]:
                self.frame_pair_indexes.append((i, j))

        self.length = len(self.optical_flow_paths)

        self._validate()

    def _validate(self):
        assert set(self.frame_paths.keys()) == set(self.pose_paths.keys()), \
            "Metadata is corrupt. Frame indexes for frame metadata did not match frame indexes for pose metadata."

        metadata_not_found_template = "The optical flow dataset references frame #{index}, " \
                                      "however the metadata for frame #{index} could not be found."

        for i, j in self.frame_pair_indexes:
            if i not in self.frame_indexes:
                raise RuntimeError(metadata_not_found_template.format(index=i))

            if j not in self.frame_indexes:
                raise RuntimeError(metadata_not_found_template.format(index=j))


class OpticalFlowDatasetBuilder:
    """
    Utility for creating the optical flow dataset for a video.

    This class handles the creation of the necessary folders, saving the files to disk and writing the metadata to
    disk. An `OpticalFlowDatasetBuilder` object can be used as a context manager in a `with` statement so that the
    metadata is automatically written to disk when you're done. If you don't use this class in this way, then make sure
    to call the `finalise()` method when you're finished making changes to the dataset to make sure the dataset's
    metadata is saved properly.
    """

    def __init__(self, base_dir: str, camera: Camera):
        assert os.path.isdir(base_dir), "Could not find or access the path {}.".format(base_dir)

        self.base_dir = os.path.abspath(base_dir)
        self.metadata_dir = os.path.join(self.base_dir, "metadata")
        self.frame_data_dir = os.path.join(self.base_dir, "frame_data")
        self.optical_flow_dir = os.path.join(self.base_dir, "optical_flow")

        self.metadata = OpticalFlowDataMetadata(self.metadata_dir)
        self.metadata.set_camera(camera)

        self.metadata.make_dirs()

        for dir_name in (self.frame_data_dir, self.optical_flow_dir):
            os.makedirs(dir_name)

    def add(self, i, j, frame_i, frame_j, optical_flow_field, valid_mask, camera_poses):
        assert frame_i.shape[:2] == frame_j.shape[:2] == optical_flow_field.shape[-2:] == valid_mask.shape[-2:], \
            "The frames, optical flow field and optical flow mask must all have the same spatial dimensions. " \
            "Got {}, {}, {} and {}." \
                .format(frame_i.shape[:2], frame_j.shape[:2], optical_flow_field.shape[:2], valid_mask.shape[:2])

        if i not in camera_poses or j not in camera_poses:
            warnings.warn("A frame pair was found where there was no camera pose given (frame indices: {}, {}), "
                          "skipping frame pair.".format(i, j))
            return

        self._add_frame(i, frame_i, camera_poses[i])
        self._add_frame(j, frame_j, camera_poses[j])
        self._add_optical_flow(i, j, optical_flow_field, valid_mask)

    def _add_frame(self, frame_index, frame, camera_pose: CameraPose):
        if frame_index not in self.metadata.frame_paths:
            frame_filename = "{:04d}.png".format(frame_index)
            pose_filename = "{:04d}.pkl".format(frame_index)

            frame_path = os.path.join(self.frame_data_dir, frame_filename)
            pose_path = os.path.join(self.frame_data_dir, pose_filename)

            cv2.imwrite(frame_path, frame)
            camera_pose.save_pkl(pose_path)

            self.metadata.add_frame(frame_index, frame_filename, pose_filename)

    def _add_optical_flow(self, i, j, optical_flow_field, valid_mask):
        optical_flow_filename = "{:04d}-{:04d}.pkl".format(i, j)
        optical_flow_path = os.path.join(self.optical_flow_dir, optical_flow_filename)

        with open(optical_flow_path, 'wb') as f:
            pickle.dump((optical_flow_field, valid_mask), f)

        self.metadata.add_optical_flow(i, j, optical_flow_filename)

    def finalise(self):
        self.metadata.save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.finalise()


class OpticalFlowDataset(Dataset):
    """
    A dataset for optimising a depth estimation network as per the paper 'Consistent Video Depth Estimation' [1].
    The dataset contains the following:

    - One camera pose for the video that the dataset is based on.
    - Many 6-tuples containing a pair of sequential video frames, camera poses for each of the frames, a dense optical
      flow field describing the pixel displacement occurring between the two frames, and a mask indicating which optical
      flow vectors are 'valid'.

    [1] Luo, Xuan, Jia-Bin Huang, Richard Szeliski, Kevin Matzen, and Johannes Kopf. "Consistent Video Depth
    Estimation." arXiv preprint arXiv:2004.15021 (2020).
    """

    def __init__(self, base_dir: str, image_transform):
        assert os.path.isdir(base_dir), \
            "The following dataset path could not be found or read: {}.".format(base_dir)

        self.base_dir = os.path.abspath(base_dir)

        self.metadata_dir = os.path.join(self.base_dir, "metadata")
        self.frame_data_dir = os.path.join(self.base_dir, "frame_data")
        self.optical_flow_dir = os.path.join(self.base_dir, "optical_flow")

        self.metadata = OpticalFlowDataMetadata(self.metadata_dir)
        self.metadata.load()

        self.image_transform = image_transform

    def __getitem__(self, index):
        i, j = self.metadata.frame_pair_indexes[index]

        optical_flow_path = os.path.join(self.optical_flow_dir, self.metadata.optical_flow_paths[i][j])
        frame_i_path = os.path.join(self.frame_data_dir, self.metadata.frame_paths[i])
        frame_j_path = os.path.join(self.frame_data_dir, self.metadata.frame_paths[j])
        pose_i_path = os.path.join(self.frame_data_dir, self.metadata.pose_paths[i])
        pose_j_path = os.path.join(self.frame_data_dir, self.metadata.pose_paths[j])

        with open(optical_flow_path, 'rb') as f:
            optical_flow, valid_mask = pickle.load(f)

        frame_i = cv2.imread(frame_i_path)
        frame_i = self.image_transform(frame_i)

        frame_j = cv2.imread(frame_j_path)
        frame_j = self.image_transform(frame_j)

        pose_i = CameraPose.load_pkl(pose_i_path)
        pose_j = CameraPose.load_pkl(pose_j_path)

        R_i = pose_i.R.as_matrix()
        R_i = R_i.astype(np.float32)

        t_i = pose_i.t
        t_i = t_i.astype(np.float32)

        R_j = pose_j.R.as_matrix()
        R_j = R_j.astype(np.float32)

        t_j = pose_j.t
        t_j = t_j.astype(np.float32)

        return frame_i, frame_j, R_i, t_i, R_j, t_j, optical_flow, valid_mask

    def __len__(self):
        return self.metadata.length


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
        return self.to_tensor(self.arrays[index])

    def __len__(self):
        return len(self.arrays)


def wrap_MiDaS_transform(x):
    return {"image": x}


def unwrap_MiDaS_transform(x):
    return x["image"]


def scale(x):
    return x / 255.0


def create_image_transform(height, width, normalise=True):
    transforms = [
        scale,
        wrap_MiDaS_transform,
        Resize(
            width,
            height,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
    ]

    if normalise:
        transforms.append(NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    transforms += [
        PrepareForNet(),
        unwrap_MiDaS_transform
    ]

    return Compose(transforms)
