---
title: "Using Nvidia GPUs in Podman containers in Fedora 37"
date: 2022-11-22
permalink: /posts/2022-11-22-fedora-podman-gpu
tags: 
    - linux 
    - docker 
    - podman

---



It's been a while since Fedora has moved away from docker to podman. I'm not going into the nitty gritty details on docker vs podman but if you really want to know more you can read this [article](https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/When-to-use-Docker-vs-Podman-A-developers-perspective).

## Source

This article has been compiled from [https://ask.fedoraproject.org/t/how-to-run-tensorflow-gpu-in-podman/8486/9](https://ask.fedoraproject.org/t/how-to-run-tensorflow-gpu-in-podman/8486/9).

## Getting started - Nvidia Driver

Make sure to have the non free RPM Fusion Nvidia driver installed before you start because `nvidia-container-toolkit` will need that driver. (*No the open source driver won't do.*)

If you don't have it already, then run the following commands to install it.

```bash
sudo rpm -Uvh http://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm

sudo rpm -Uvh http://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

# install the driver
# as per : https://rpmfusion.org/Howto/NVIDIA

sudo dnf update -y
sudo dnf install -y xorg-x11-drv-nvidia
```

Wait for a while (the kernel needs to run mods for the drivers) and then type `reboot` in the terminal and hit enter.

## Install Podman

Podman should be installed by default but there's no harm in checking!

```bash
podman --version
```

## Check the CUDA version on host

Running `nvidia-smi` should suffice.

```bash
nvidia-smi 
   
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 520.56.06    Driver Version: 520.56.06    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:2D:00.0  On |                  N/A |
|  0%   39C    P8    27W / 420W |   1035MiB / 24576MiB |      3%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      7451      G   /usr/libexec/Xorg                 380MiB |
|    0   N/A  N/A      7651      G   /usr/bin/gnome-shell              205MiB |
|    0   N/A  N/A      7958      G   ...mail/        4MiB |
|    0   N/A  N/A      8904    C+G   ...933855740524873253,131072      155MiB |
|    0   N/A  N/A     15775      G   ...e/Steam/ubuntu12_32/steam      101MiB |
|    0   N/A  N/A     16072      G   ...ef_log.txt --shared-files      183MiB |
+-----------------------------------------------------------------------------+
```

## Install the nvidia-container-toolkit

Nvidia officially provides conatiner toolkit releases for RHEL but not for Fedora. However, since it's compatible with RHEL, Fedora should be too. So what you have to do here is use the release for RHEL. Use the latest release of RHEL (which is now 9.x).

```bash
distribution=rhel9.0

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
```

Now you can install it .

```bash
sudo dnf clean expire-cache \
   && sudo dnf install -y nvidia-container-toolkit
```

## Configuration tweaks to run rootless containers

By default the container toolkit requires that you run GPU containers as root. This isn't ideal and can open up security issues in many cases. Running as a user process should be the better alternative.

```bash
sudo sed -i 's/^#no-cgroups = false/no-cgroups = true/;' /etc/nvidia-container-runtime/config.toml
```

## Does it work?

There are a few ways to check. We can run some gpu specific code inside or, just call nvidia-smi inside a podman container. So for a container, we need an image. You can find cuda images on docker hub:https://hub.docker.com/r/nvidia/cuda/ . Visit the `tags` section and pick the one that matches the cuda version on your host OS (check `nvidia.smi` output from earlier).

Let's try with `nvidia-smi` .

```bash
podman run --rm --security-opt=label=disable \
    --hooks-dir=/usr/share/containers/oci/hooks.d/ \
    nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

When asked to select a registry, select the docker.io one. If you see something similar the the output I after running a container, your installation was successful. However, if you get some error at this point, it'll most likely be an issue with the driver (especially if you're still on the open source GPU drivers) or a cuda version mismatch.

Now why do we need that `--security-opt=label=disable` flag? Because SELinux will block your container from accessing the GPU otherwise.

```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 520.56.06    Driver Version: 520.56.06    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:2D:00.0  On |                  N/A |
|  0%   38C    P8    26W / 420W |    993MiB / 24576MiB |      6%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                              
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

Now you can try defining some Dockerfile with the code you want to run on your GPU and it should work fine. You can also try the sample workload container image from Nvidia.

## Done!

That's kinda it!
