import warnings

import cv2
import numpy as np


class VideoData:
    def __init__(self, frames, fps, height, width):
        self.frames = np.array(frames)
        self.fps = fps
        self.height = height
        self.width = width

    @property
    def num_frames(self):
        return len(self.frames)

    @property
    def shape(self):
        return self.height, self.width


def read_video(video_path, logger):
    input_video = cv2.VideoCapture(video_path)
    if not input_video.isOpened():
        raise RuntimeError("Could not open video from the path {}.".format(video_path))
    logger.log("Opened video from path the {}.".format(video_path))
    frames = []

    while input_video.isOpened():
        has_frame, frame = input_video.read()

        if not has_frame:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        logger.log("Extracted {:,d} video frames.\r".format(len(frames)), end="")

    print()

    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.log("Frame dimensions: {}x{}.".format(width, height))

    fps = input_video.get(cv2.CAP_PROP_FPS)
    logger.log("Frames per second: {}.".format(fps))

    if input_video.isOpened():
        input_video.release()

    return VideoData(frames, fps, height, width)


def write_video(frames, video_data, video_output_path, logger):
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    video_writer = cv2.VideoWriter(video_output_path, fourcc, video_data.fps, (video_data.width, video_data.height))

    if frames.shape[-1] == 1:
        warnings.warn("Was given input with one colour channel, stacking frames to get three channels.")
        frames = np.concatenate([frames] * 3, axis=-1)

    assert frames.shape[-1] == 3, \
        "Video must be in NHWC format and be RGB (i.e. C=3). Got {} shaped input.".format(frames.shape)
    assert frames.dtype == np.uint8, "Frames must be uint8, got {}.".format(frames.dtype)

    for i, frame in enumerate(frames):
        video_writer.write(frame)
        logger.log("Wrote {}/{} frames.\r".format(i + 1, len(frames)), end="")

    print()

    video_writer.release()
    logger.log("Wrote video to {}.".format(video_output_path))
