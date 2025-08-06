"""
Utilities for embedding
"""
import sys
sys.path.append('../common')
import config
from datetime import datetime
import cv2
import math
import os
import re
import numpy as np
import glob
import pickle
from subprocess import PIPE, run
import statistics
from colorama import Fore, Back, Style
from numbers_parser import Document


###################################
#   TRIAL PREP/MANAGEMENT UTILS   #
###################################
def prepare_dirs():
    os.system(f"rm {config.trial_materials_path}/config_slm.py")
    os.system(f"rm {config.trial_materials_path}/config_3d_slm.py")
    os.system(f"rm {config.trial_materials_path}/ber_logging.txt")
    os.system(f"rm {config.trial_materials_path}/adaptation_logging.csv")
    os.system(f"rm {config.trial_materials_path}/adaptation_logging.numbers")
    os.system(f"rm {config.trial_materials_path}/imgs/*")
    os.system(f"rm -r {config.trial_materials_path}/bmps/*")
    os.system(f"rm {config.trial_materials_path}/cell_signal_plots/*")
    os.system(f"rm {config.trial_materials_path}/signal_pkls/*")
    os.system(f"rm {config.trial_materials_path}/vids_overtime/*")
    os.system(f"rm {config.trial_materials_path}/cellvids_overtime/*")
    os.system(f"rm {config.trial_materials_path}/synchronization/*")
    os.system(f"rm {config.trial_materials_path}/payloads/*")
    os.system(f"rm {config.trial_materials_path}/features/*")
    os.system(f"rm {config.trial_materials_path}/perceptibility/*")

    os.makedirs(f"{config.trial_materials_path}/imgs", exist_ok = True)
    os.makedirs(f"{config.trial_materials_path}/bmps", exist_ok = True)
    os.makedirs(f"{config.trial_materials_path}/cell_signal_plots", exist_ok = True)
    os.makedirs(f"{config.trial_materials_path}/signal_pkls", exist_ok = True)
    os.makedirs(f"{config.trial_materials_path}/vids_overtime", exist_ok = True)
    os.makedirs(f"{config.trial_materials_path}/cellvids_overtime", exist_ok = True)
    os.makedirs(f"{config.trial_materials_path}/synchronization", exist_ok = True)
    os.makedirs(f"{config.trial_materials_path}/payloads", exist_ok = True)
    os.makedirs(f"{config.trial_materials_path}/features", exist_ok = True)
    os.makedirs(f"{config.trial_materials_path}/perceptibility", exist_ok = True)

def write_config_summary():
    summary_file = open(f"{config.trial_materials_path}/config_summary.txt", "w")

    summary_file.write(f"============================= CONFIG {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =============================\n")
    if config.TESTING_MODE:
        summary_file.write(f"TESTING_MODE = {config.TESTING_MODE}\n")
        summary_file.write(f"Chosen ID and Dynamic Hash K = {config.identity_hash_k}\n")
    else:
        summary_file.write(f"TESTING_MODE = {config.TESTING_MODE}\n")    
    if not config.ADAPT:
        summary_file.write(f"ADAPT = {config.ADAPT}\n")
    else:
        summary_file.write(f"ADAPT = {config.ADAPT}\n")
    summary_file.write(f"Dynamic Hash k = {config.dynamic_hash_k}, Identity Hash k = {config.identity_hash_k}\n")
    summary_file.write(f"Frequency = {config.frequency}\n")
    summary_file.write(f"Target lower frequency range = {config.lower_range_start}-{config.lower_range_end}\n")
    summary_file.write(f"Target upper frequency range = {config.upper_range_start}-{config.upper_range_end}\n")
    summary_file.write(f"Target frequency epsilon = {config.target_lower_epsilon}-{config.target_upper_epsilon}\n")
    summary_file.write(f"Fade = {config.fade}\n")
    summary_file.write(f"N = {config.N}, buffer_space = {config.buffer_space}\n")
    summary_file.write(f"Max cells_H = {config.max_cells_H}, Max cells_W = {config.max_cells_W}\n")
    summary_file.write(f"Num border cells = {config.num_pilot_cells}, Num info cells = {config.num_info_cells}\n")
    summary_file.write(f"Max bits per window = {config.max_bits}\n")
    if config.rs_n <= config.rs_k:
        summary_file.write(f"Error corrector: Soft Viterbi. No room for RS encoding\n")
    else:
        summary_file.write(f"Error corrector: Concatenated Viterbi + RS encoding with n = {config.rs_n}, k = {config.rs_k}\n")
    summary_file.write(f"Viterbi payload size =  {config.viterbi_payload_size}\n")
    summary_file.write(f"Initial SLM BGR = {config.init_slm_bgr}\n")
    summary_file.write(f"Initial min tot SLM = {config.init_min_tot_slm}\n")
    summary_file.write("=================================================================")
    summary_file.close()

def stash_trial_data(stash_path):
    """
    Stash relevant trial data -- currently only the payloads directory and config_summary -- at <stash_path>
    Meant to be called prior to prepare_dirs, which will clear out any data in the trial materials directory, if any
    """
    # find any folders starting with "trial_materials_"
    trial_materials_folders = glob.glob("trial_materials_*")
    if len(trial_materials_folders) != 0:
        trial_materials_folder_name = trial_materials_folders[0] # just the first if there are for some reason more than one
    else:
        return 
    
    if os.path.exists(f"{stash_path}/{trial_materials_folder_name}"):
        i = 2
        while True:
            if os.path.exists(f"{stash_path}/{trial_materials_folder_name}_{i}"):
                i += 1
            else:
                out_trial_materials_folder_name = f"{trial_materials_folder_name}_{i}"
                break
    else:
        out_trial_materials_folder_name = trial_materials_folder_name
    
    print("Stashing trial data at: ", trial_materials_folder_name)
    os.makedirs(f"{stash_path}/{out_trial_materials_folder_name}", exist_ok = True)
    os.system(f"cp -r {trial_materials_folder_name}/payloads {stash_path}/{out_trial_materials_folder_name}")
    os.system(f"cp -r {trial_materials_folder_name}/signal_pkls {stash_path}/{out_trial_materials_folder_name}")
    os.system(f"cp {trial_materials_folder_name}/config_summary.txt {stash_path}/{out_trial_materials_folder_name}")
    os.system(f"cp {trial_materials_folder_name}/adaptation_logging.csv {stash_path}/{out_trial_materials_folder_name}")
    os.system(f"rm -r {trial_materials_folder_name}")

def create_slm_config():
    out = open(f"{config.trial_materials_path}/config_slm.py", "w")
    out.write('"""This file is a modified copy of config.py generated for the SLM and copied to the Pi at the start of the session."""\n')
    out.write(f"fade = {config.fade}\n")
    out.write(f"scale_factor_disp = {config.scale_factor_disp}\n")
    out.write(f"N = {config.N}\n")
    out.write(f"buffer_space = {config.buffer_space}\n")
    out.write(f"slm_W = {config.slm_W}\n")
    out.write(f"slm_H = {config.slm_H}\n")
    out.write(f"slm_W = {config.slm_W}\n")
    out.write(f"localization_N = {config.localization_N}\n")
    out.write(f"barcode_window_duration  = {config.barcode_window_duration}\n")
    out.write(f"localization_frequency  = {config.localization_frequency}\n")
    out.write(f"frequency = {config.frequency}\n")
    out.write(f"target_channel = {config.target_channel}\n")
    template_content = open("../common/config_slm_template.py", "r").read()
    out.write(template_content)
    out.close()

def create_slm_3d_config():
    out = open(f"{config.trial_materials_path}/config_3d_slm.py", "w")
    out.write('"""This file is a modified copy of config.py generated for the SLM and copied to the Pi at the start of the session."""\n')
    out.write(f"fade = {config.fade}\n")
    out.write(f"scale_factor_disp = {config.scale_factor_disp}\n")
    out.write(f"N = {config.N}\n")
    out.write(f"buffer_space = {config.buffer_space}\n")
    out.write(f"slm_W = {config.slm_W}\n")
    out.write(f"slm_H = {config.slm_H}\n")
    out.write(f"slm_W = {config.slm_W}\n")
    out.write(f"localization_N = {config.localization_N}\n")
    out.write(f"barcode_window_duration  = {config.barcode_window_duration}\n")
    out.write(f"row_freqs  = {config.row_freqs}\n")
    out.write(f"outline_width  = {config.outline_width}\n")
    out.write(f"freqs_lcm  = {config.freqs_lcm}\n")
    out.write(f"target_channel = {config.target_channel}\n")
    template_content = open("../common/config_slm_3d_template.py", "r").read()
    out.write(template_content)
    out.close()


##########################
#   LOG CREATION UTILS   #
##########################
    
def nice_log_from_csv(csv_path):
    csv_name = csv_path.split("/")[-1].split(".csv")[0]
    output_name = f"{config.trial_materials_path}/{csv_name}.numbers"
    doc = Document()
    sheets = doc.sheets
    table = sheets[0].tables[0]
    loc_tyle = doc.add_style(
            font_color = (0, 0, 200)
    )
    f = open(csv_path, "r")
    row_num = 0
    for line in f:
        cell_vals = line.split(",")
        if cell_vals[15] != "-": #this is a localization marker row
            style = loc_tyle
        else:
            style = None
        
        for i, cell_val in enumerate(cell_vals):
            if type(cell_val) == float or type(cell_val) == np.float64 or type(cell_val) == np.float32 or type(cell_val) == np.float16:
                table.write(row_num, i, f"{cell_val:.3f}", style=style)
            else:
                table.write(row_num, i, cell_val, style=style)

        row_num += 1

    doc.save(output_name)
    
def get_curr_row(doc_file_name):
    doc = Document(doc_file_name)
    sheets = doc.sheets
    tables = sheets[0].tables
    rows = tables[0].rows()
    for i, row in enumerate(rows):
        if row[0].value is None or row[0].value == "":
            return i
    return 0

def write_log_row(doc_file_name, row_data, row_num = None, font_rgb = None):
    if not os.path.exists(doc_file_name):
        doc = Document() # create and save new document with this name 
        doc.save(doc_file_name)
        
    doc = Document(doc_file_name)
    if font_rgb is not None:
        if type(font_rgb) == list:
            font_rgb = tuple(font_rgb)
        style = doc.add_style(
            font_color = font_rgb
        )
    else:
        style = None

    sheets = doc.sheets
    table = sheets[0].tables[0]
    if row_num is None:
        row_num = get_curr_row(doc_file_name)
    for i, cell_val in enumerate(row_data):
        if type(cell_val) == float or type(cell_val) == np.float64 or type(cell_val) == np.float32 or type(cell_val) == np.float16:
            table.write(row_num, i, f"{cell_val:.3f}", style=style)
        else:
            table.write(row_num, i, str(cell_val), style=style)
    doc.save(doc_file_name)


##########################
# IMAGE SEQUENCE HANDLING #
##########################


def set_enc_params(override_slm_W = None, override_slm_H = None, 
               override_N = None, override_buffer_space = None, 
               override_localization_N = None, override_reserved_localization_cells = None, 
               override_target_channel = None, override_frequency = None, 
               override_barcode_window_duration = None):
    if override_N is not None:
        N = override_N
    else:
        N = config.N
    
    if override_buffer_space is not None:
        buffer_space = override_buffer_space
    else:
        buffer_space = config.buffer_space
    
    if override_localization_N is not None:
        localization_N = override_localization_N
    else:
        localization_N = config.localization_N
    
    if override_reserved_localization_cells is not None:
        reserved_localization_cells = override_reserved_localization_cells
    else:
        reserved_localization_cells = config.reserved_localization_cells
    
    if override_slm_H and override_slm_W:
        slm_W, slm_H = override_slm_W, override_slm_H
    else:
        slm_W, slm_H = config.slm_W, config.slm_H
    
    if override_target_channel is not None:
        target_channel = override_target_channel
    else:
        target_channel = config.target_channel

    if override_frequency is not None:
        frequency = override_frequency
    else:
        frequency = config.frequency
    
    if override_barcode_window_duration is not None:
        barcode_window_duration = override_barcode_window_duration
    else:
        barcode_window_duration = config.barcode_window_duration

    max_cells_W = int((slm_W ) / (N + buffer_space))
    max_cells_H = int((slm_H + buffer_space) / (N + buffer_space))

    if localization_N is not None:
        max_cells = max_cells_H * max_cells_W - 4*localization_N**2
    else:
        max_cells = max_cells_H * max_cells_W 


    return slm_W, slm_H, N, buffer_space, localization_N, max_cells_W, max_cells_H, reserved_localization_cells, target_channel, frequency, barcode_window_duration, max_cells


def get_imgseq_info(start_time, end_time, image_timestamps, fpss):
    """
    Determined indices of first image (start_image_timestamp_i) and last image (end_image_timestamp_i)
    in the sequence. Also return the effective FPS of the core unit camera during this sequence
    """
    start_time_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
    start_image_timestamp_i = image_timestamps.index(min(image_timestamps, key=lambda b: abs(start_time_dt - datetime.strptime(b, "%Y-%m-%d %H:%M:%S.%f")))) - 10
    end_time_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S.%f")
    end_image_timestamp_i = image_timestamps.index(min(image_timestamps, key=lambda b: abs(end_time_dt - datetime.strptime(b, "%Y-%m-%d %H:%M:%S.%f")))) + 7
    fps = statistics.mean(fpss[start_image_timestamp_i:end_image_timestamp_i + 1])
    return fps, start_image_timestamp_i, end_image_timestamp_i

def delete_imgseq(start_image_timestamp_i, end_image_timestamp_i, image_dir):
    for i in range(start_image_timestamp_i, end_image_timestamp_i + 1): 
        img_path = "{}/hom_{}.png".format(image_dir, i)
        npy_path = "{}/{}.npy".format(image_dir, i)
        os.system(f"rm {img_path}")
        os.system(f"rm {npy_path}")

def get_imgseq(start_image_timestamp_i, end_image_timestamp_i):
    adaptive_log(f"Start and end image timestamps of sequence: {start_image_timestamp_i} {end_image_timestamp_i}", "DEBUG")
    img_seq = []
    for i in range(start_image_timestamp_i, end_image_timestamp_i + 1): #+3 for good measure
        img = cv2.imread(f"{config.trial_materials_path}/imgs/hom_{i}.png")
        if config.colorspace == "ycrcb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        img_seq.append(img)
    img_seq = np.array(img_seq)
    return img_seq


##########################
# IMAGE PROCESSING #
##########################

def overall_cell_to_row_col(overall_cell_num):
    """
    return row + col index of cell number, taking into account any adjustments that need to be made if 
    localization markers are being used
    """

    if config.localization_N is None:
        r = overall_cell_num // config.max_cells_W
        c = overall_cell_num % config.max_cells_W
    else:
        # adjust row and column positions by considering offsets introduced by localization markers
        if overall_cell_num < (config.max_cells_W - config.localization_N*2) * config.localization_N: 
            # cells occuring in rows in between top two markers
            r = overall_cell_num // (config.max_cells_W - config.localization_N*2)
            c = (overall_cell_num % (config.max_cells_W - config.localization_N*2)) + config.localization_N
        elif overall_cell_num >=  (config.max_cells_W - config.localization_N*2) * config.localization_N + (config.max_cells_H - 2*config.localization_N)*config.max_cells_W:
            # cells occuring in rows in between bottom two markers
            for i in range(config.localization_N):
                if overall_cell_num < (config.max_cells_W - config.localization_N*2) * config.localization_N + (config.max_cells_H - 2*config.localization_N)*config.max_cells_W + (i+1)*(config.max_cells_W - config.localization_N*2):
                    r = (overall_cell_num + 2*config.localization_N**2 + i*(config.localization_N*2)) // config.max_cells_W
                    c = (overall_cell_num + 2*config.localization_N**2 + config.localization_N + i*(config.localization_N*2)) % config.max_cells_W
                    break
        else: #cells occuring in normal rows, not between markers
            r = (overall_cell_num + 2*config.localization_N**2)// config.max_cells_W
            c = (overall_cell_num + 2*config.localization_N**2) % config.max_cells_W
    return r, c


def hom_images(Hom, start_image_timestamp_i, end_image_timestamp_i):
    for i in range(start_image_timestamp_i, end_image_timestamp_i + 1): #include end_imgage_timestamp_i
        img_path = "{}/{}.npy".format(f"{config.trial_materials_path}/imgs", i)
        # if not os.path.exists(img_path):
        #     while not os.path.exists(img_path):
        #         print(f"File {img_path} doesn't exist! Sleeping a bit until it appears..")
        #         time.sleep(0.01)
        # img = cv2.imread(img_path)
        img = np.load(img_path)
        img = cv2.warpPerspective(img, Hom, (640, 360))
        cv2.imwrite(f"{config.trial_materials_path}/imgs/hom_{i}.png", img)

def vis_img_seq(start_image_timestamp_i, end_image_timestamp_i):
    for i in range(start_image_timestamp_i, end_image_timestamp_i + 1):
        img = cv2.imread(f"{config.trial_materials_path}/imgs/{i}.png")
        cv2.imshow("Image sequence", img)
        if cv2.waitKey(1) == ord('q'): break

def save_img_seq(start_image_timestamp_i, end_image_timestamp_i, output_name, fps, input_colorspace = "bgr", vidcodec = 'mp4v', Hom=False, save_cells=False, cells_output_prefix=None):
    if vidcodec == 'mp4v':
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    elif vidcodec == 'rgba':
        fourcc = cv2.VideoWriter_fourcc('R', 'G', 'B', 'A')
    else:
        adaptive_log("Invalid requested video codec in save_img_seq", "ERROR")

    dummy = np.load("{}/{}.npy".format(f"{config.trial_materials_path}/imgs", start_image_timestamp_i))
    H, W, _ = dummy.shape
    if vidcodec == "mp4v":
        if Hom:
            out = cv2.VideoWriter("{}.mp4".format(output_name), fourcc, fps, (640, 360))
        else:
            out = cv2.VideoWriter("{}.mp4".format(output_name), fourcc, fps, (W, H))
    else:
        if Hom:
            out = cv2.VideoWriter("{}.avi".format(output_name), fourcc, fps, (640, 360))
        else:
            out = cv2.VideoWriter("{}.avi".format(output_name), fourcc, fps, (W, H))
    for i in range(start_image_timestamp_i, end_image_timestamp_i + 1): #include end_imgage_timestamp_i
        if Hom:
            img = cv2.imread("{}/hom_{}.png".format(f"{config.trial_materials_path}/imgs", i))
        else:
            img = np.load("{}/{}.npy".format(f"{config.trial_materials_path}/imgs", i))
        if input_colorspace == "ycrcb":
            img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
        
        if Hom: #draw cell crops for nice visualization
            
            for overall_cell_num in range(config.max_cells):
                cell_row, cell_col = overall_cell_to_row_col(overall_cell_num)
                cell_top = cell_row*(config.N + config.buffer_space) 
                cell_left = cell_col*(config.N + config.buffer_space) 
                cell_bottom = cell_top + config.N
                cell_right = cell_left + config.N
                img = cv2.rectangle(img, (cell_left, cell_top), (cell_right, cell_bottom), 1)

        out.write(img)
    out.release()
    
    if save_cells: #save a separate video for each cell
        N = config.N 
        buffer_space = config.buffer_space
        if config.corner_markers:
            offset = N
        else:
            offset = 0
        
        for cell_row in range(config.max_cells_H):
                for cell_col in range(config.max_cells_W):
                    print("Saving vid for r, c: ", cell_row, cell_col)
                    cell_top = offset + cell_row*(N+buffer_space) 
                    cell_left = offset + cell_col*(N+buffer_space) 
                    cell_bottom = cell_top + N
                    cell_right = cell_left + N
                    out = cv2.VideoWriter("{}_r{}_c{}.avi".format(cells_output_prefix, cell_row, cell_col), fourcc, fps, (N, N))
                    for i in range(start_image_timestamp_i, end_image_timestamp_i + 1): #include end_imgage_timestamp_i
                        img = cv2.imread("{}/hom_{}.png".format(config.temp_img_dir, i))
                        if input_colorspace == "ycrcb":
                            img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
                        cell = img[cell_top:cell_bottom + 1, cell_left:cell_right + 1, :]
                        out.write(cell)
                    out.release()


################
# ARDUCAM CAMERA UTILS #
################

def get_uvc_cam_id(camera_name):
    command = ["uvc-util/uvc-util", "--list-devices"]
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    cam_id = 0
    for item in result.stdout.splitlines():
        if camera_name in item:
            cam_id = int(item.split(" ")[0])

    adaptive_log("{} UVC  cam id is {}".format(camera_name, cam_id), "DEBUG")
    return cam_id 

def get_cam_id(camera_name):
    """
    Get camera index corresponding to the desired camera connected to the host computer
    https://forum.opencv.org/t/how-to-specify-exact-camera-by-id-to-prevent-two-usb-cameras-from-being-switched-around-macos/12351/6
    """
    command = ['ffmpeg','-f', 'avfoundation','-list_devices','true','-i','""']
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    cam_id = 0

    for item in result.stderr.splitlines():
        if "AVFoundation audio devices" in item:
            break
        if camera_name in item:
            cam_id = int(item.split("[")[2].split(']')[0])
    
    adaptive_log("{} cam id is {}".format(camera_name, cam_id), "DEBUG")
    return cam_id

def load_uvccam_settings(camera_name):
    print("Loading UVC cam settings.")

    if not os.path.exists("curr_uvccam_settings.pkl"):   
        return

    with open("curr_uvccam_settings.pkl", "rb") as pklfile:
        settings = pickle.load(pklfile)

    target_settings = {"white-balance-temp", "exposure-time-abs", "auto-focus", "gain", "saturation", "auto-exposure-mode", "sharpness", "contrast", "hue", "gamma", "auto-white-balance-temp", "focus-abs"}

    cam_id = get_uvc_cam_id(camera_name)
    
    # if auto-white-balance-temp or auto-exposure-mode are options here, need to toggle them off 
    # before setting exposure or white-bal, and thus these values must be loaded first. 
    # this can be enforced by sorting alphabetically, which of course puts controls 
    # that start with "auto" first
    settings = dict(sorted(settings.items()))
    for key, value in settings.items():
        if key not in target_settings:
            continue
        if type(value) == str:
            command = ["uvc-util/uvc-util", "-I", f"{cam_id}", "-s", f'{key}="{value}"']
        else:
            command = ["uvc-util/uvc-util", "-I", f"{cam_id}", "-s", f'{key}={value}']
        result = run(command, stdout=PIPE, stderr=PIPE)
    print("Finished loading UVC cam settings.")

def save_uvccam_settings(camera_name):
    """
    Retrieves current Arducam settings and saves to pkl file
    """
    cam_id = get_uvc_cam_id(camera_name)

    #get list of controls
    command = ["uvc-util/uvc-util", "-I", f"{cam_id}", "--list-controls"]
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    #get and save all control values
    controls = []
    for item in result.stdout.splitlines():
        controls.append(item[2:].replace("\n", ""))
    settings = {}
    for control in controls:
        command = ["uvc-util/uvc-util", "-I", f"{cam_id}", "-g", control]
        result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        if result.stdout == '':
            continue
        
        val = result.stdout.split("= ")[1].replace("\n", "")
        if "{" in val:
            curr_control_val = val
        else:
            if "false" in val: #can't just do bool conversion 
                curr_control_val = False
            elif "true" in val:
                curr_control_val = True
            else:
                curr_control_val = int(val)
        settings[control] = curr_control_val
    
    print("Saving following settings")
    print(settings)
    with open("curr_uvccam_settings.pkl", "wb") as pklfile:
        pickle.dump(settings, pklfile)
  

def confirm_cam_settings(camera_name):
    load_uvccam_settings(camera_name)

    cam_id = get_cam_id(camera_name) #cam id and uvc_cam_id can be different. one is for interacing over uvc, one is for actually fetching frames
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Display with current settings. Press [a] to accept. Press [s] to overwrite with new ones.', frame)
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('s'): 
            save_uvccam_settings(camera_name)
            break
        elif pressed_key == ord('a'): 
            break


    cap.release()
    for i in range(10):
        cv2.destroyAllWindows()
    adaptive_log("Confirmed camera settings.", "INFO")

    

    
##########################
# OTHER #
##########################
    
def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

def parse_last_disp_info(message):
    last_disp_seq = re.search(r'(?<=Last displayed sequence: )(.*)(?=. Start)', message)
    if last_disp_seq is None:
        return
    else:
        last_disp_seq = int(last_disp_seq.group())
    start_time = re.search(r'(?<=Start: )(.*)(?=. End)', message).group()
    end_time = re.search(r'(?<= End: )(.*)(?=. Timestamps)', message).group()
    timestamps = re.search(r'(?<= Timestamps: )(.*)(?=. READY FOR METADATA)', message).group()
    timestamps = timestamps.split(';')[:-1]
    return last_disp_seq, start_time, end_time, timestamps
 
def adaptive_log(message, log_level, flags = []):
    if log_level == "DEBUG":
        if config.LOG_LEVEL == "DEBUG":
            output = "ADAPTIVE [DEBUG]: {}".format(message)
            for f in flags:
                output = f + output
            output += Style.RESET_ALL
            print(output)
    elif log_level == "INFO":
        if config.LOG_LEVEL == "DEBUG" or config.LOG_LEVEL == "INFO":
            output = "ADAPTIVE  [INFO]: {}".format(message)
            for f in flags:
                output = f + output
            output += Style.RESET_ALL
            print(output)
    elif log_level == "WARNING":
        if config.LOG_LEVEL == "DEBUG" or config.LOG_LEVEL == "INFO" or config.LOG_LEVEL == "WARNING":
            print(Fore.WHITE + Back.YELLOW + "ADAPTIVE [WARNING]: {}".format(message) + Style.RESET_ALL)
    elif log_level == "ERROR":
        print(Fore.YELLOW + Back.RED + "ADAPTIVE [ERROR]: {}".format(message) + Style.RESET_ALL)
  