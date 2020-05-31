FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER hirakawat


# basic packages
RUN apt-get -y update && apt-get -y upgrade && \
        apt-get install -y sudo cmake g++ gfortran \
        libhdf5-dev pkg-config build-essential \
        wget curl git htop tmux vim ffmpeg rsync openssh-server \
        libsm6 libxext6 libxrender-dev libglib2.0-0 \
        python3 python3-dev libpython3-dev && \
        apt-get -y autoremove && apt-get -y clean && apt-get -y autoclean && \
        rm -rf /var/lib/apt/lists/*

# cuda path
ENV CUDA_ROOT /usr/local/cuda
ENV PATH $PATH:$CUDA_ROOT/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$CUDA_ROOT/lib64:$CUDA_ROOT/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib
ENV LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:/usr/local/cuda/lib$LIBRARY_PATH


# python3 modules
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
        pip3 install --upgrade --no-cache-dir wheel six setuptools cython numpy scipy==1.4.1 \
        matplotlib seaborn scikit-learn scikit-image pillow requests opencv-python plac \
        jupyterlab networkx h5py pandas plotly protobuf tqdm tensorboardX colorama setproctitle && \
        pip install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/cu90/torch_stable.html
