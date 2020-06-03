import plac
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from MiDaS.models.midas_net import MidasNet
from Video3D.dataset import NumpyDataset, create_image_transform
from Video3D.io import write_video, read_video
from utils.tools import TimerBlock


def inference(video_data, model_path, logger, scale_output=False, batch_size=8):
    model = MidasNet(model_path, non_negative=False)
    model = model.cuda()
    model.eval()
    logger.log("Loaded model weights from {}.".format(model_path))

    transform = create_image_transform(video_data.height, video_data.width)

    video_dataset = NumpyDataset(video_data.frames, transform)
    data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        num_frames_processed = 0
        depth_maps = []

        for batch_i, batch in enumerate(data_loader):
            images = batch.cuda()
            depth = model(images)
            depth = F.interpolate(depth.unsqueeze(1), size=(video_data.height, video_data.width), mode='bilinear',
                                  align_corners=True)
            depth_maps.append(depth.detach().cpu())

            num_frames_processed += images.shape[0]

            logger.log("Generated {}/{} depth maps.\r".format(num_frames_processed, video_data.num_frames), end="")

    print()

    depth_maps = torch.cat(depth_maps, dim=0)

    if scale_output:
        depth_maps = (255 * (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min())).to(torch.uint8)

    depth_maps = depth_maps.permute((0, 2, 3, 1)).numpy()

    return depth_maps


@plac.annotations(
    video_path=plac.Annotation("The path to the input video.", kind="option", type=str, abbrev="i"),
    model_path=plac.Annotation("The path to the depth estimation model weights.", kind="option", type=str, abbrev="m"),
    video_output_path=plac.Annotation("The path to the write the output to.", kind="option", type=str, abbrev="o"),
)
def main(video_path, model_path, video_output_path):
    with TimerBlock("Load Video") as block:
        video_data = read_video(video_path, block)

    with TimerBlock("Depth Estimation") as block:
        depth_maps = inference(video_data, model_path, block, scale_output=True, batch_size=4)
        write_video(depth_maps, video_data, video_output_path, block)


if __name__ == '__main__':
    plac.call(main)
