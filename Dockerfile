FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt update && \
    apt install -y git \
                   vim && \
    rm -rf /var/lib/apt/lists