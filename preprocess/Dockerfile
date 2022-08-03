FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

# Prepare and empty machine for building
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    vim \
    wget \
    unzip \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
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

# Build and install ceres solver
RUN apt-get -y install \
    libatlas-base-dev \
    libsuitesparse-dev
RUN git clone https://github.com/ceres-solver/ceres-solver.git --branch 1.14.0
RUN cd ceres-solver && \
	mkdir build && \
	cd build && \
	cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
	make -j4 && \
	make install

# Build and install COLMAP

# Note: This Dockerfile has been tested using COLMAP pre-release 3.6.
# Later versions of COLMAP (which will be automatically cloned as default) may
# have problems using the environment described thus far. If you encounter
# problems and want to install the tested release, then uncomment the branch
# specification in the line below
RUN git clone https://github.com/colmap/colmap.git #--branch 3.6

RUN cd colmap && \
	git checkout 96d4ba0b55c0d1f98c8c432420ecd6540868c398 && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make -j4 && \
	make install

# Install MiniConda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda init bash

# TODO: use git clone when released
ADD ../../ml-neuman /neuman

RUN cd /neuman/preprocess  && \
    git clone --recurse-submodules https://github.com/compphoto/BoostingMonocularDepth.git && \
    cd BoostingMonocularDepth && \
    git checkout ecedd0c0cf5e1807cdab1c5154351a97168e710d

RUN cd /neuman/preprocess  && \
    git clone --recurse-submodules https://github.com/jiangwei221/ROMP.git && \
    cd ROMP && \
    git checkout f1aaf0c1d90435bbeabe39cf04b15e12906c6111

RUN cd /neuman/preprocess  && \
    git clone --recurse-submodules https://github.com/jiangwei221/detectron2.git && \
    cd detectron2 && \
    git checkout 2048058b6790869e5add8832db2c90c556c24a3e

RUN cd /neuman/preprocess  && \
    git clone --recurse-submodules https://github.com/jiangwei221/mmpose.git && \
    cd mmpose && \
    git checkout 8b788f93200ce6485e885da0c736f114e4de8eaf

# ROMP environment
RUN . /root/.bashrc && \
    conda create -n ROMP python==3.8.8 && \
    conda activate ROMP && \
    conda install -c pytorch pytorch=1.9.1 torchvision=0.10.1 cudatoolkit=10.2 && \
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath && \
    conda install -c bottler nvidiacub && \
    conda install pytorch3d -c pytorch3d

RUN . /root/.bashrc && \
    conda activate ROMP && \
    cd /neuman/preprocess/ROMP && \
    pip install -r requirements.txt && \
    pip install av

# detectron2 environment
RUN . /root/.bashrc && \
    conda activate ROMP && \
    cd /neuman/preprocess && \
    python -m pip install -e detectron2 && \
    pip install setuptools==59.5.0

# mmpose environment
RUN . /root/.bashrc && \
    conda create -n open-mmlab python=3.7 -y && \
    conda activate open-mmlab && \
    conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch && \
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html && \
    cd /neuman/preprocess/mmpose && \
    pip install -r requirements.txt && \
    pip install -v -e . && \
    pip install openmim && \
    mim install mmdet

# NeuMan environment
RUN . /root/.bashrc && \
    conda create -n neuman_env python=3.7 -y && \
    conda activate neuman_env && \
    conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch && \
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath && \
    conda install -c bottler nvidiacub && \
    conda install pytorch3d -c pytorch3d && \
    conda install -c conda-forge igl && \
    pip install opencv-python joblib open3d imageio tensorboardX chumpy lpips scikit-image ipython matplotlib
