# Running VeriLight core unit

## Prep
### Connections and Power
1. Plug the power jack into the SLM.
2. When you see the SLM has projected the striped splash screen, plug the USB-C power cord into the Raspberry Pi.
3. Plug in the core unit camera's USB to your computer.
4. Plug in the other end of the Ethernet cord to your computer. 


### Software
Raspberry Pi: 
1. Open a separate terminal and SSH into the Raspberry Pi. Note that it may take a while after you plug in the Raspberry Pi for it to start up and be ready for SSH connections.
2. Make sure the Raspberry Pi is connected to WiFi. This is a very important step because the embedding code requires global timestamps to synchronize between the Rasperry Pi and controlling computer. To check if connected to WiFi, run `ping 8.8.8.8`. If it outputs `64 bytes received...` it is connected. In the case it is not, run the command
`sudo raspi-config` on the Pi, and then use the GUI to connect to WiFi. Eventually, it will connect to WiFi (sometimes can be quite slow). You will know it has when ping 8.8.8.8 returns "64 bytes..." instead of "ICMP ..."

    Note: Sometimes the connection over Ethernet is finnicky. In this case, you can connect to the Raspberry Pi via WiFi. To do this, ensure that the Raspberry Pi is connected to WiFi (using the steps above). Then find the public IP of the Pi by using the command
    `sudo ifconfig`. The public IP will be listed in the wlan0 section of the output.
    Edit the host_ip variable in config to be that IP 
3. Disable desktop display on the Pi by running `sudo systemctl stop lightdm.service`. (Based on [this](https://www.makeuseof.com/how-to-disable-lightdm-linux/#:~:text=To%20disable%20LightDM%2C%20all%20you,if%20you're%20using%20runit.)) article. Note it is important to run this before next step.
4. Enter the directory dlpdldcr230npevm_python_support_software_1.0 on the Pi and run `python3 init_parallel_mode.py`.
5. Activate the Pi-specific conda environment (see [getting_started.md](getting_started.md) if you haven't already created this environment). 

Your computer:
1. Open two terminals for running code locally. Enter the [embedding](../embedding) directory on each terminal, as all scripts you will later need to run locally are located here. 
2. Activate the provided conda environment (created from [df_mac.yml](df_mac.yml)) in each.


## Running  
You will need three terminals open: the first SSH'd into the Raspberry Pi, and the second two for running scripts locally on your computer, which should be ready to go from the Environment Prep section. 

### Calibration
1. In your Raspberry Pi terminal, start the position calibration light server by running `python pos_light_server.py`
2. In your first local terminal, start the position calibration code by running `python pos_calibration.py`.
The first thing that it will do is display the core unit camera's video feeed and ask you to verify that it looks ok in terms of color quality and focus. If it looks ok, click in the camera display and then press the "a" key. If the color or focus are look off (which may happens the first time that you run the calibration or deployment code for the first time after plugging in the core unit camera), keep the view open and follow the instructions below to adjust the settings. When they look ok, click in the camera display and then press the "s" key to save these settings to ensure they are used in future runs. 
3. Once you click accept, the calibration process will start. The idea of calibration is for the system to learn where on the projection surface it should look for each specific cell. To do this, the SLM will project a calibration sequence, consisting of four red cells - one in each corner - blinking repeatedly. Once it's done projecting this sequence, your local computer will start processing the video from the core unit camera and eventually pop up several display windows to guide you through the process of accepting the calibration settings. First, it will show you its "Inferred calibration corners." The terminal prompt wil ask you if you would like to accept this output ("Are the inferred calibration corners ok? y/n"). If the calibration went well, you should see four very bright squares, corresponding to the blinking calibration corners, and a black background. The squares should be labeled - from top to bottom, left to right: 1, 2, 3, 4. If you see this, enter "y" in the terminal prompt. Otherwise, enter "n" and then one by one, from top to bottom, left to right, click the centers of the correct corners. When you are done, click in the display and enter "q". 

The calibration will now perform the homography (mapping from core unit camera view to reference SLM view). It will display the source (a SLM calibration BMP template), destination (the produced heatmap), and result (heatmap warped so that the calibration corners in it match the calibration corners in the template). If the homography was successfully, you will see that the warped heatmap (result) corners match up very closely with the template corners. Press "q" on your computer to close each of these three displays. Once they are all closed, the system pos_calibration.py will create (if this is your first time running calibration) or update the file [embedding/curr_hom.pkl](embedding/curr_hom.pkl) and then terminate. If the homography looked successfull and you see that curr_hom_pkl has been updated, you are good to go! 

Once you have run the calibration, you cannot move the camera! Even a small change in its position or angle relative to the projection surface could really 

### Deployment
The real-time embedding system consists of three main components, distributed across three scripts. 
(1) [driver.py](../embedding/driver.py) runs on the computer, accepting new digests produced by (2) [digest_process.py](../embedding/digest_process.py) and making them into bitmaps for the SLM to display, and also coordinating adaptive embedding. (3) [light_server.py](../embedding/light_server.py) is run on the Raspberry Pi and accepts new bitmap sequences for the SLM to display. From your setup, it should be located in the home directory of the Rasperry Pi. 

1. In your Raspberry Pi terminal, start the light server by running `python light_server.py`.
2. In your second local terminal, start the digest extraction process by running `python digest_process.py`. It will set up the face detection and MediaPipe extractor and then hang waiting for indication from the driver.py that it should start. Let it hang while you start driver.py below.
3. In your first local terminal, start the driver by running `python driver.py`. The first thing it will do is copy the config file to the Raspberry Pi to ensure that all code running on each machine is using the same embedding parameters. Then, it will display the core unit camera's video feed. Using the same procedure described in the Calibration section, accept or update+save the camera settings. 

Start all the data collection recording devices, and then enter "y" in the driver terminal's prompt.
After a few seconds, you will see the SLM indicate that is has received a window's sequence of bitmaps. Shortly after it will begin to (visibly) display them.
The first few windows (each a few seconds long) will be visible, and then eventually adaptation will kick in and the code will become "imperceptible." Once this happens, wait for a few seconds (say 3), and then tell the person that they can start reading the script. 

### Adjusting Core Unit Camera Settings
To adjust the camera settings, you will have to use uvc-util script provided in the uvc-util Git submodule. Run the following commands in a new terminal: 
- `./uvc-util/uvc-util -I 0 -s auto-focus=False`
- `./uvc-util/uvc-util -I 0 -s focus-abs=<x>`, where x is a value between 30 and 100 that makes the camera view seem the most focused. You can experiment with different values until it seems in focus.
- `./uvc-util/uvc-util -I 0 -s auto-white-balance-temp=True` 
- `./uvc-util/uvc-util -I 0 -s auto-white-balance-temp=False`
After setting focsus-abs and auto-white-balance-temp, you should see changes in the camera view. 


## Shutdown
1. Run `sh shutdown.sh` on Pi, WAIT until green indicator light on Pi is off, then unplug Pi, then unplug the SLM.
2. Unplug everything else however you'd like. 
