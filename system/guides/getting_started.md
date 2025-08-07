
# Getting started with VeriLight

## Building a core unit
The original VeriLight prototype uses a [DLPDLCR230NPEVM](https://www.ti.com/tool/DLPDLCR230NPEVM#tech-docs) SLM, a [Raspberry Pi 4B](https://www.amazon.com/Raspberry-Pi-RPI4-MODBP-4GB-Model-4GB/dp/B09TTNF8BT?tag=googhydr-20&source=dsa&hvcampaign=electronics&gbraid=0AAAAA-b0EosazeVwcGR7W9PfISTaVP6m3&gclid=CjwKCAjw5PK_BhBBEiwAL7GTPQVOYkEATCAGZLEQKGejSL8wZ14tbyIgVWfgBJjCokSiEwoTJYMl3hoC9D8QAvD_BwE) as an interface to the SLM, and a PC (Macbook) as the central processor running digest extraction, building bitmaps for display, and sending those to the SLM Raspberry Pi via a socket.

This guide is specific to these components, though the framework is compatible with alternatives. 

### Parts
- [Raspberry Pi 4B](https://www.amazon.com/Raspberry-Pi-RPI4-MODBP-4GB-Model-4GB/dp/B09TTNF8BT?tag=googhydr-20&source=dsa&hvcampaign=electronics&gbraid=0AAAAA-b0EosazeVwcGR7W9PfISTaVP6m3&gclid=CjwKCAjw5PK_BhBBEiwAL7GTPQVOYkEATCAGZLEQKGejSL8wZ14tbyIgVWfgBJjCokSiEwoTJYMl3hoC9D8QAvD_BwE) + SD card as described in manual
- [DLPDLCR230NPEVM](https://www.ti.com/tool/DLPDLCR230NPEVM#tech-docs) SLM + power supply as described in manual
- Any UVC-compatible USB RGB camera. We use [this](https://www.arducam.com/product/8mp-imx179-autofocus-usb-camera-module-with-waterproof-protection-case-for-windows-linux-android-and-mac-os/) Arducam
- A PC capable of SSHing into the RaspberryPi

### Assembly
The physical assembly of the core unit from the above parts is mostly up to the user. The only requirement is that the USB camera should be fixed relative to the SLM after calibration. You can achieve this however you want. We laser print a base to hold the SLM, Raspberry Pi, and Arducam steady. For interested users, we provide the base's design as both a [.beam file](raspi_and_evm230NP_base.beam) (for use with Beamo laser cutters) and [an SVG](raspi_and_evm230NP_base.svg).

## Software and Hardware Setup
1. Clone this repo. 
2. Clone the submodules by running:
```
git submodule init
git submodule update
```

### Hardware
1. Use [Raspberry Pi Imager](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up/2) to flash an SD card with a legacy 64-bit version*. Edit the configuration settings to set username and password and enable SSH, so that you can SSH with Pi in headless mode from even the first boot. 
2. Eject the SD card and plug it back in to computer. It should show up as a device named 'bootfs'. Replace config.txt on this device with the config.txt file provided in [dlpdlcr230np_python_support_code](dlpdlcr230np_python_support_code). Note: Modifying config.txt from the Pi after booting up did NOT work. Only modifying it on the SD card as mentioned above did. 
3. Put SD card in Raspberry Pi.
4. Connect DLP230NP EVM to Raspberry Pi via provided FPC cable.
5. Connect Pi to personal computer via Ethernet cable. The Pi should be accessible at raspberrypi.local, by default.

\* Otherwise, I only see a black screen. Similarly reported (and resolved) [here](https://e2e.ti.com/support/dlp-products-group/dlp/f/dlp-products-forum/1288278/dlpdlcr230npevm-init_parallel_mode-displays-black-screen). I have validated with Raspbian 11 (bullseye) 64-bit.

### Software
#### On your PC 
Note: These instructions assume your PC is a Macbook. 
1. Create a conda environment from the [provided yaml](df_mac.yml):
`conda env create -f df_mac.yaml`
2.  Enter the uvc-util submodule (in the [embedding folder](../embedding/)) and build the uvc-util tool as described in the README. Note: you will need xcode to do this. The VeriLight code expects a binary named uvc-util at the top level of the uvc-util folder.
<!-- 3. Download the UltraLightFace detector labels and checkpoint and place them in the [Ultra-Light-Fast-Generic-Face-Detector-1MB](../common/Ultra-Light-Fast-Generic-Face-Detector-1MB/) folder. Note: included in submodule-->
3. Optionally enable SSH/SCP to the RaspberryPi from your PC without a password. Assuming there is already a keypair in ~/.ssh, just run `ssh-copy-id <user>@<host>` (e.g., `ssh-copy-id verilight@raspberrypi.local`)

#### On the Raspberry Pi
1. SSH into the RaspberryPi from your PC.
2. [Install conda](https://github.com/conda-forge/miniforge#download)
3. Create a conda environment with Python 3.8 for running the code. Activate it and install the following necessary packages:
    ```
    pip install opencv-python
    pip install colorama
    pip install numbers_parser
    ```
Note that this is a different conda environment than the one provided for your Mac, and is already located on the Raspberry Pi.
4. Install xterm: `sudo apt install xterm`
5. Copy the entire [dlpdlcr230np_python_support_code](dlpdlcr230np_python_support_code) directory to the home directory of the Raspberry Pi.
6. Copy [light_server.py](../embedding/light_server.py), [pos_light_server.py](../embedding/pos_light_server.py), [pi_display_utils.py](../embedding/pi_display_utils.py),  [embedding_utils.py](../embedding/embedding_utils.py), [psk_encode_minimal.py](../embedding/psk_encode_minimal.py), and [black.png](../embedding/black.png) to the home directory of the Raspberry Pi.
7. Add following lines to  *~/.bashrc* on Pi:
    ```
    echo -e '\e]11;rgb:00/00/00\a'sud
    echo -e '\e]10;rgb:00/00/00\a'
    ```
    These make background and text of terminal black, which is necessary for our encoding. Otherwise, the terminal will appear in between BMP displays.
8. Disable screen blanking via raspi-config.
   "Display Options" > "Screen Blanking" and select "No" to disable the feature
9. Copy/move the following files to the Raspberry Pi:
    - [shutdown.sh](shutdown.sh)
    - [pos_light_server.py](../embedding/pos_light_server.py)
    - [light_server.py](../embedding/light_server.py)
    - [pi_display_utils.py](../embedding/pi_display_utils.py)


## Sanity check your work
Make sure you can operate the SLM to project things. Either of the below methods will work.

### Try using as Desktop mirror
1. Enter the [dlpdlcr230np_python_support_code](dlpdlcr230np_python_support_code) on the Pi. Run `python3 init_parallel_mode.py`. Your Pi Desktop should now be projected.

### Try projecting a video in full-screen mode
1. Disable desktop display on the Pi by running `sudo systemctl stop lightdm.service`. (Based on [this](https://www.makeuseof.com/how-to-disable-lightdm-linux/#:~:text=To%20disable%20LightDM%2C%20all%20you,if%20you're%20using%20runit.)) article. Note it is important to run this before next step.
2. Enter [dlpdlcr230np_python_support_code](dlpdlcr230np_python_support_code) on the Pi. Run `python3 init_parallel_mode.py`.
3. In one terminal, run `sudo xinit `. CAUTION: Do not run this before running init_parallel_mode.py. This appears to cause xinit to crash soon after starting, with the only fix being a reboot. 
4. In other terminal, `export DISPLAY=:0`. Now you can view a video using vlc, opencv, etc. 
5. To shut down, first run shutdown.sh on Pi, WAIT until green indicator light on Pi is off, then unplug Pi, then unplug DLP230NP EVM.


