# Ubuntu 18.04
1.  Ensure NVIDIA GPU drivers are installed - `nvidia-smi` is the quickest way to check.
2.  Install Docker.
3.  Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
4.  Download checkpoint file(s) from the links listed in the [README file](README.md).
5.  Run `./launch_docker.sh` to build the Docker image and launch the Docker container.
6.  Run `./install.sh` from within Docker container to install the custom neural network layers.
7.  You can now run the code. 
    **Note:** You have to have use the command `python3` to access Python.
    -   Example command:
        ```shell script
        python3 main.py --inference --model FlowNet2 --save_flow --inference_dataset ImagesFromFolder --inference_dataset_root /path/to/frames --inference_dataset_iext jpeg --resume /path/to/checkpoints/FlowNet2_checkpoint.pth.tar 
        ``` 
        Adjust the last three parameters as needed.
        
        This will spit out `.flo` files under the `work/inference/run.epoch-0-flow-field` directory.

# Ubuntu 20.04
There were issues installing nvidia-docker on the latest version of Ubuntu. 
It appears some drivers were not available.

# Windows
The latest version of Docker (19.03) does not seem to support GPU on Windows yet.