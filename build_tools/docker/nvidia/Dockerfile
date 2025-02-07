# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for building IREE using CMake and testing IREE with NVIDIA GPUs.

# To use the host GPUs, `docker run` must be called with the `--gpus all` flag.

# We use .deb files that we host because we have to pin the version exactly to
# match the host machine and packages routinely dissapear from the Ubuntu
# apt repositories.
ARG NVIDIA_GL_DEB="libnvidia-gl-460_460.39-0ubuntu0.18.04.1_amd64.deb"
ARG NVIDIA_COMPUTE_DEB="libnvidia-compute-460_460.39-0ubuntu0.18.04.1_amd64.deb"
ARG NVIDIA_COMMON_DEB="libnvidia-common-460_460.39-0ubuntu0.18.04.1_all.deb"


FROM gcr.io/iree-oss/base@sha256:61f948eb15c90596b9faabbca168de2c7d5008e5d2c0b9c5c26a0efe572198d6 AS fetch-nvidia
ARG NVIDIA_COMMON_DEB
ARG NVIDIA_GL_DEB
ARG NVIDIA_COMPUTE_DEB

WORKDIR /fetch-nvidia
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_COMMON_DEB}"
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_GL_DEB?}"
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_COMPUTE_DEB}"


# Set up the image and working directory by inheriting the vulkan CMake
# configuration.
# Note that we don't start from NVIDIA's docker base:
# - nvidia/cuda (https://hub.docker.com/r/nvidia/cuda):
#     it's.. for CUDA.
# - nvidia/vulkan (https://hub.docker.com/r/nvidia/vulkan):
#      does not support Ubuntu 18.04.
# This allows to share configuration with base CMake, but it also means we need
# to MATCH the driver version between the host machine and the docker image.
FROM gcr.io/iree-oss/base@sha256:61f948eb15c90596b9faabbca168de2c7d5008e5d2c0b9c5c26a0efe572198d6 AS final
ARG NVIDIA_COMMON_DEB
ARG NVIDIA_GL_DEB
ARG NVIDIA_COMPUTE_DEB

COPY --from=fetch-nvidia \
  "/fetch-nvidia/${NVIDIA_COMMON_DEB}" \
  "/fetch-nvidia/${NVIDIA_GL_DEB}" \
  "/fetch-nvidia/${NVIDIA_COMPUTE_DEB}" \
  /tmp/

RUN apt-get install "/tmp/${NVIDIA_COMMON_DEB?}" \
  "/tmp/${NVIDIA_GL_DEB?}" \
  "/tmp/${NVIDIA_COMPUTE_DEB?}"
