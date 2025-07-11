# ref: https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/2.0.1-cuda11.8-ubuntu22.04/Dockerfile
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities.
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg libsm6 libxext6 build-essential wget vim tmux \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && sudo wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip && sudo gunzip /usr/local/bin/ninja.gz && sudo chmod +x /usr/local/bin/ninja

# Create a working directory mounted at the startup
RUN mkdir /home/kirillz
WORKDIR /home/kirillz

# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /home/kirillz
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
    && chmod -R 777 $HOME

COPY requirements.txt .

RUN sudo apt-get install python3-dev python3-venv -y && \
    python3 -m venv ~/venv && \
    . ~/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    echo "source ~/venv/bin/activate" >> ~/.bashrc && \
    echo "alias sai='sudo apt install'\nalias pi='pip install'" >> ~/.bashrc

RUN mkdir -p /home/kirillz/neural_terrain_representation/data

COPY . .

RUN . ~/venv/bin/activate && pip install -e .