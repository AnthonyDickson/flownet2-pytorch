import os
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plac
from matplotlib import gridspec

from utils.flow_utils import flow2img


@plac.annotations(
    first_frame_index=plac.Annotation(
        "The index of the first frame.",
        type=int, kind="positional"
    ),
    second_frame_index=plac.Annotation(
        "The index of the second frame.",
        type=int, kind="positional"
    ),
    video_path=plac.Annotation(
        'The path to the source video file.',
        type=str, kind='option', abbrev='i'
    ),
    workspace_path=plac.Annotation(
        "Where to save and load intermediate results to and from.",
        type=str, kind="option"
    ),
)
def visualise_optical_flow(first_frame_index, second_frame_index, video_path: str, workspace_path: str = "workspace/"):
    """Visualise the optical flow between two video frames from a generated optical flow dataset."""
    video_name = Path(video_path).stem
    base_path = os.path.join(workspace_path, video_name)
    frame_pair = (first_frame_index, second_frame_index)
    optical_flow_path = os.path.join(base_path, "dataset", "optical_flow", "{:04d}-{:04d}.pkl".format(*frame_pair))

    with open(optical_flow_path, 'rb') as f:
        optical_flow, valid_mask = pickle.load(f)

    frame_path_fmt = os.path.join(base_path, "frames", "{:04d}.png")
    frame = cv2.cvtColor(cv2.imread(frame_path_fmt.format(frame_pair[0])), cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(cv2.imread(frame_path_fmt.format(frame_pair[1])), cv2.COLOR_BGR2RGB)

    optical_flow_viz = flow2img(optical_flow.transpose((1, 2, 0)))
    optical_flow_viz *= valid_mask.transpose((1, 2, 0))
    optical_flow_viz[optical_flow_viz == 0] = 255

    gs = gridspec.GridSpec(4, 4)

    ax1 = plt.subplot(gs[:2, :2])
    ax1.imshow(frame)
    ax1.set_title("First Frame")

    ax2 = plt.subplot(gs[:2, 2:])
    ax2.imshow(frame2)
    ax2.set_title("Second Frame")

    ax3 = plt.subplot(gs[2:, 1:3])
    ax3.imshow(optical_flow_viz)

    X, Y = np.meshgrid(np.arange(0, frame.shape[1], 1), np.arange(0, frame.shape[0], 1))
    U, V = optical_flow * valid_mask
    every_n = 32
    ax3.quiver(X[::every_n, ::every_n], Y[::every_n, ::every_n], U[::every_n, ::every_n], V[::every_n, ::every_n])
    ax3.set_title("Optical Flow")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plac.call(visualise_optical_flow)
