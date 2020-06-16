import os
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
import plac
from open3d.open3d_pybind.camera import PinholeCameraIntrinsic
from open3d.open3d_pybind.geometry import RGBDImage, PointCloud, TriangleMesh, Image
from open3d.open3d_pybind.io import write_point_cloud, write_triangle_mesh

from Video3D.io import read_video
from utils.tools import TimerBlock


def generate_point_cloud(camera_intrinsics, frame, depth_map, output_path, frame_i, num_frames, logger,
                         depth_scale=0.01, depth_trunc=10000):
    color = Image(frame).flip_vertical()
    depth = Image(depth_map.transpose((1, 2, 0))).flip_vertical()

    rgbd = RGBDImage.create_from_color_and_depth(color=color,
                                                 depth=depth,
                                                 depth_scale=depth_scale, depth_trunc=depth_trunc,
                                                 convert_rgb_to_intensity=False)
    logger.log("Created RGBD frame {}/{}.".format(frame_i + 1, num_frames))

    point_cloud = PointCloud.create_from_rgbd_image(image=rgbd, intrinsic=camera_intrinsics)
    logger.log("Created point cloud {}/{}.".format(frame_i + 1, num_frames))
    reflection_along_z_axis = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    point_cloud = point_cloud.transform(reflection_along_z_axis)
    logger.log("Reflected point cloud along z-axis for frame {}/{}.".format(frame_i + 1, num_frames))
    point_cloud.estimate_normals()
    logger.log("Estimated normals for point cloud for frame {}/{}.".format(frame_i + 1, num_frames))


    point_cloud_filename = "{:04d}.ply".format(frame_i)
    point_cloud_path = os.path.join(output_path, point_cloud_filename)
    write_point_cloud(filename=point_cloud_path, pointcloud=point_cloud)
    logger.log("Saved point cloud for frame {}/{} to {}.".format(frame_i + 1, num_frames, point_cloud_path))
    return point_cloud


def generate_mesh(point_cloud, output_path, frame_i, num_frames, logger):
    # mesh, _ = TriangleMesh.create_from_point_cloud_poisson(pcd=point_cloud)
    distances = point_cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = max(0.5, 1.5 * avg_dist)
    mesh = TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd=point_cloud,
        radii=o3d.utility.DoubleVector([radius, 2 * radius])
    )

    logger.log("Created triangle mesh {}/{}.".format(frame_i + 1, num_frames))
    mesh_filename = "{:04d}.ply".format(frame_i)
    mesh_path = os.path.join(output_path, mesh_filename)
    write_triangle_mesh(filename=mesh_path, mesh=mesh)
    logger.log("Saved triangle mesh for frame {}/{} to {}.".format(frame_i + 1, num_frames, mesh_path))

    return mesh

@plac.annotations(
    video_path=plac.Annotation(
        'The path to the source video file.',
        type=str, kind='option', abbrev='i'
    ),
    workspace_path=plac.Annotation(
        "Where to save and load intermediate results to and from.",
        type=str, kind="option"
    )
)
def main(video_path, workspace_path="workspace"):
    print(dict(
        video_path=video_path,
        workspace_path=workspace_path,
    ))

    video_name = Path(video_path).stem
    workspace_path = os.path.join(os.path.abspath(workspace_path), video_name)
    colmap_path = os.path.join(workspace_path, "parsed_colmap_output.pkl")

    point_cloud_path = os.path.join(workspace_path, "point_cloud")
    point_cloud_path_before = os.path.join(point_cloud_path, "before")
    point_cloud_path_after = os.path.join(point_cloud_path, "after")

    mesh_path = os.path.join(workspace_path, "mesh")
    mesh_path_before = os.path.join(mesh_path, "before")
    mesh_path_after = os.path.join(mesh_path, "after")

    assert os.path.isfile(video_path), "Could not open video located at {}.".format(video_path)
    assert os.path.isdir(workspace_path), "Could not find the workspace folder {}.".format(workspace_path)
    assert os.path.isfile(colmap_path), "Could not open the parsed COLMAP output at {}.".format(colmap_path)

    if not os.path.exists(point_cloud_path):
        os.makedirs(point_cloud_path)
        os.makedirs(point_cloud_path_before)
        os.makedirs(point_cloud_path_after)

    if not os.path.exists(mesh_path):
        os.makedirs(mesh_path)
        os.makedirs(mesh_path_before)
        os.makedirs(mesh_path_after)

    with TimerBlock("Load Data") as block:
        video_data = read_video(video_path, block, convert_to_rgb=True)

        dnn_depth_map_path = os.path.join(workspace_path, "dnn_depth_map.npy")
        dnn_depth_maps = np.load(dnn_depth_map_path, mmap_mode='r')
        block.log("Loaded unoptimised depth maps from {}.".format(dnn_depth_map_path))

        optimised_dnn_depth_map_path = os.path.join(workspace_path, "optimised_dnn_depth_map.npy")
        optimised_dnn_depth_maps = np.load(optimised_dnn_depth_map_path, mmap_mode='r')
        block.log("Loaded optimised depth maps from {}.".format(optimised_dnn_depth_map_path))

        with open(colmap_path, 'rb') as f:
            camera, _, _ = pickle.load(f)

        block.log("Loaded camera intrinsics from {}.".format(colmap_path))

        camera_intrinsics = PinholeCameraIntrinsic(width=camera.width, height=camera.height, fx=camera.focal_length,
                                                   fy=camera.focal_length, cx=camera.center_x, cy=camera.center_y)

    with TimerBlock("Generate Point Clouds from RGBD") as block:
        for frame_i, (frame, dnn_depth, optimised_dnn_depth) in enumerate(
                zip(video_data, dnn_depth_maps, optimised_dnn_depth_maps)):
            # TODO: Parallelise this.
            block.log("Creating point cloud and mesh for unoptimised depth estimation model.")
            point_cloud = generate_point_cloud(camera_intrinsics, frame, dnn_depth, point_cloud_path_before, frame_i, len(video_data), block)
            generate_mesh(point_cloud, mesh_path_before, frame_i, len(video_data), block)

            block.log("Creating point cloud and mesh for optimised depth estimation model.")
            point_cloud = generate_point_cloud(camera_intrinsics, frame, optimised_dnn_depth, point_cloud_path_after, frame_i, len(video_data), block)
            # TODO: This is too slow with ball pivot algorithm. Make it faster.
            # generate_mesh(point_cloud, mesh_path_after, frame_i, len(video_data), block)


if __name__ == '__main__':
    plac.call(main)
