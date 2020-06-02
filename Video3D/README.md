# Video3D
This package implements the paper "Consistent Video Depth Estimation" [1]. 
For the monocular depth estimation network I use MiDaS [2]. This project also relies on FlowNet2 [3] and I use a public 
PyTorch implementation [4].
## Program Pipeline Overview
1.  Take video.
2.  Run reconstruction algorithm on video with COLMAP.
3.  Load video and COLMAP output into main routine of this package (`python -m Video3D`).
4.  Extract camera intrinsics, camera poses per frame and sparse depth maps per frame from COLMAP output.
5.  Generate dense depth maps using a pretrained monocular depth estimation network.
6.  Calculate the relative depth scaling factor *S* and adjust the camera poses accordingly.
7.  Sample frame pairs from the video.
8.  For each frame pair:
    1.  Align the frames to remove most of the movement.
    2.  Generate dense optical flow for the frame pair.
    3.  Filter frame pairs based on the consistency of the optical flow.
9.  Create a dataset from the frame pairs and optical flow fields.
10. [TODO] Perform test-time training on the monocular depth estimation network.
11. [TODO] Generate depth maps for the video using the trained network. 

## Getting Started
1.  Get the docker container running. See [INSTALL.md](../INSTALL.md) for more details.
2.  Start the docker container using [launch_docker.sh](../launch_docker.sh).
3.  Run the main script. Run `python3 -m Video3D -h` for help on the command line arguments.
    - Example command: `python3 -m Video3D data/asakusa_29fps/colmap/ -i data/asakusa_29fps/source.webm -d checkpoints/model.pt -f checkpoints/FlowNet2_checkpoint.pth.tar`

### Caching
By default, the script will cache intermediate results since they are usually too large to fit in memory. You can clear the cache if need be by simply removing the cache folder: `rm -r .cache/`.
## References
1. Luo, Xuan, Jia-Bin Huang, Richard Szeliski, Kevin Matzen, and Johannes Kopf. "Consistent Video Depth Estimation." arXiv preprint arXiv:2004.15021 (2020).
2. Lasinger, Katrin, René Ranftl, Konrad Schindler, and Vladlen Koltun. "Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer." arXiv preprint arXiv:1907.01341 (2019). https://github.com/intel-isl/MiDaS
3. Ilg, Eddy, Nikolaus Mayer, Tonmoy Saikia, Margret Keuper, Alexey Dosovitskiy, and Thomas Brox. "Flownet 2.0: Evolution of optical flow estimation with deep networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2462-2470. 2017. https://github.com/lmb-freiburg/flownet2
4. Reda, Fitsum, Robert Pottorff, Jon Barker, and Bryan Catanzaro. "flownet2-pytorch: Pytorch implementation of flownet 2.0: Evolution of optical flow estimation with deep networks." (2017). https://github.com/NVIDIA/flownet2-pytorch