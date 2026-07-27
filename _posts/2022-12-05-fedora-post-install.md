---
title: "Fedora post installation steps"
date: 2022-12-05
permalink: /posts/2022-12-05-fedora-post-install
tags: 
    - linux
    - fedora

---

I use Fedora on my workstation to make my home and lab computers consistent with each other. This is a collection post installation steps I had followed. If you plan to use Fedora sometime in the future and are looking for a guide, you can use this one as a reference.

*Please keep in mind that the steps mentioned here work on the current release version, which is 40. Fedora has a 6 months release cycle so it's kinda difficult to write a guide for every new release. However, I've been following these steps since Fedora 33 with very minor changes so these shouldn't differ much in future releases.*

## Update system

The first thing to do after installing every linux distro: update your system.

```bash
sudo dnf update
```

## Add Windows to grub for dual boot (For dual booting)

Fedora uses grub2 and it works differently than on ubuntu or other distros. You can't just run grub update and call it a day.

Run `os-prober` to check for other os entries.

```bash
sudo os-prober
```

Update grub entries

```bash
sudo grub2-mkconfig -o /etc/grub2.cfg
sudo grub2-mkconfig -o /etc/grub2-efi.cfg
```

## Edit grub timeout

Edit `/etc/default/grub` and change `GRUB_TIMEOUT` to the desired value. The default is 5 (5 seconds). (Edit with sudo) Once done, update the grub entries again (commands above).

## Additional software repos

### rpm-fusion

There are two repos to enable. The one with free software and the other with proprietary software. If you happen to have an Nvidia GPU or some hardware part that needs proprietary driver, you'll need the second one. Why not the free Nvidia driver? Because it's bad. Real bad. [Apple Maps bad](https://youtu.be/tVq1wgIN62E). Period.

```bash
sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
```

### Setting up CUDA for Nvidia GPUs

```bash
sudo dnf install xorg-x11-drv-nvidia
```

Once installed, reboot and check with `nvidia-smi`.

*Note that this does not come with `cudnn` or `nvcc`.* To have those, refer to this guide on [rpmfusion](https://rpmfusion.org/Howto/CUDA).

### Nvidia Drivers with a Secure Boot enabled system

If you happen to have secure boot enabled in the UEFI / BIOS, first create a key entry for secure boot and then install the Nvidia driver, otherwise, your system will fall back to the open source driver. Please follow the instructures in the rpmfusion guide to set up the key entries for secure boot : [here](https://rpmfusion.org/Howto/Secure%20Boot).

### Flathub

A lot of applications are packaged as Flatpaks in Fedora. Especially if you switch to something like Fedora Silverblue or Kinoite, everything is a Flatpak. It's enabled by default in Fedora, only that you've to add Flathub as a source of Flatpak apps. (Flathub gives you access to Discord, Signal, Telegram etc.)

```bash
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo

flatpak update
```

## Make dnf faster

Unlike apt or zypper or pacman, dnf doesn't look for the fastest mirror to download packages from by deafult. You've to explicitly tell it to do so.

Add the following lines to your `/etc/dnf/dnf.conf` file. (Edit with sudo)

```plaintext
fastestmirror=True
# change this number to 5 or 10 if you don't have a very fast   
# internet connection
max_parallel_downloads=20 
defaultyes=True
```

## Improved fonts

Fedora maintainers have a strict policy of shipping only free and open source material with the distro. For this, you may miss out on some proprietary fonts used in many websites. Luckily you can enable them with this fedora community package package or copr package.

```bash
sudo dnf install mscore-fonts-all -y
```

## Command Line Utilities

Which ones to install? Well, that's upto you. I personally prefer [btop](https://github.com/aristocratos/bpytop) and htop for monitoring the system and nvidia-smi to check the GPU usage over time.

```bash
sudo dnf install -y btop htop
```

To run nvidia-smi in a loop:

```bash
# refereshes every 1 second
watch -n 1.0 nvidia-smi
```

Or you can use nvtop

```bash
sudo dnf install -y nvtop
```

## Default Shell

The default shell on most linux distros is bash and Fedora is no exception here. I personally prefer zsh with [oh-my-zsh](https://github.com/ohmyzsh/ohmyzsh) and [spaceship-prompt](https://github.com/denysdovhan/spaceship-prompt). Some people like the fish shell. It's completely upto you actually. Pick the one that better suits your workflow.

## Enable night light

If you've never used this thing yet on other operating systems you should now. You'll appreciate it in the long run. On Gnome : go to *Settings &gt; Displays &gt; Night Light* and set it up. On KDE, you can search for Night Light and it'll show you the appropriate setting.

## Display Scaling

Fedora for somewhat reason doesn't support fractional scaling for Gnome. If you have a high resolution display, your best bet is to change the font scaling from the Tweaks app from above. Go *Tweaks &gt; Fonts &gt; Scaling Factor* and pick an appropriate value. The default is 1.00. Depending on your screen resolution, choose the one that you find most comfortable.On KDE, you can do that from `Settings -> Hardware -> Display Configuration -> Global Scale`.

## Writing Bangla

Install OpenBanglaKeyboard - https://openbangla.github.io/ Don't forget to install the fonts from https://www.omicronlab.com/bangla-fonts.html !

To install fonts, you can either use [Font Manager](https://flathub.org/apps/details/org.gnome.FontManager) in Gnome or the System Font Settings on KDE.

## Extras (advanced)

You better know what you're doing before you run anything in this section! :P

### Enable SSH

```bash
sudo dnf install -y openssh-server

sudo systemctl start sshd.service

sudo systemctl enable sshd.service
```

#### SSH Security

If you're planning to expose your system to be connected from external networks via SSH, you may want to harden the security of your system. You can follow this guide from [CTT](https://christitus.com/ssh-guide/).

### Containers

Fedora being the upstream for RHEL, prefers that you use Podman instead of Docker. However, you can install both and get your work done.

```bash
sudo dnf install -y podman
```

Instructions on installing docker can be found [here](https://docs.docker.com/engine/install/fedora/).

### NodeJS

If you need it!

```bash
sudo dnf install nodejs
node -v # shows version when installed
```

[https://developer.fedoraproject.org/tech/languages/nodejs/nodejs.html](https://developer.fedoraproject.org/tech/languages/nodejs/nodejs.html)

### Dropdown Terminals

Install guake(Gnome) or yakuake(KDE). The keyboard binding to enable it is F12. (Make sure you launch it first after installing.)

```bash
sudo dnf install guake
# or, for kde
sudo dnf install yakuake
```

## Acknowledgements

You can check these links for further tweak and post install instructions

* https://fosspost.org/things-to-do-after-installing-fedora-33/
    
* https://www.dedoimedo.com/computers/fedora-32-essential-tweaks.html
    
* https://mutschler.eu/linux/install-guides/fedora-post-install/
    
* https://eftalor.medium.com/11-things-to-do-after-installing-fedora-33-f68751eef156
    
* https://fedoramagazine.org/
    
* https://www.reddit.com/r/Fedora/
    
* https://www.youtube.com/watch?v=RrRpXs2pkzg
