FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
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
        matplotlib seaborn scikit-learn scikit-image pillow requests plac \
        opencv-python==3.4.2.17 opencv-contrib-python==3.4.2.17 open3d \
        jupyterlab networkx h5py pandas plotly protobuf tqdm tensorboardX colorama setproctitle && \
        pip install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/cu90/torch_stable.html

# COLMAP
## Dependencies
RUN apt-get -y update && apt-get -y install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev

## Ceres solver
RUN echo deb http://cz.archive.ubuntu.com/ubuntu bionic main universe >> /etc/apt/sources.list && \
    apt-get -y update && \
    apt-get -y install libatlas-base-dev libsuitesparse-dev libeigen3-dev && \
    cd / && \
    git clone https://ceres-solver.googlesource.com/ceres-solver && \
    cd ceres-solver && \
    git checkout $(git describe --tags) && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make && \
    make install

## Install COLMAP from source
RUN cd / && \
    git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    git checkout dev && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

## Cleanup
RUN rm -rf ~/.cache/pip/* && \
    apt-get -y autoremove && \
    apt-get -y clean && \
    apt-get -y autoclean && \
    rm -rf /var/lib/apt/lists/*