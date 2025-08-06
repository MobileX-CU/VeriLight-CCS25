"""
Verify a video
"""
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from create_heatmap import create_localization_heatmap
import cv2
from scipy.signal import find_peaks
from colorama import Fore, Style
import sys
import hmac
import hashlib
from dynamic_hash_check import get_dynamic_hash_dist
sys.path.append('../common/')
import config
from digest_extraction import VideoDigestExtractor
from decoding_utils import loadVideo, get_homography, get_loc_marker_center, valid_r_c, get_nonloc_cell_signal_from_imgseq, butter_bandpass_filter
from decode_sequence import decode_sequence
from bitstring_utils import bytes_to_bitstring
from rp_lsh import hamming
sys.path.append('../embedding/')
from calibration_utils import  get_user_points, detect_heatmap_cells, order_calibration_code_corners
from psk_encode_minimal import create_sample_frame


def vis_portion(i, main_sync_signal, bp_main_pred_interwin, plot_title = None, display = False, save_path = None):
    fig, axes = plt.subplots(2, tight_layout=True, figsize = (12, 6))
    axes[0].plot(main_sync_signal)
    if i == len(bp_main_pred_interwin) - 1:
        axes[0].vlines([bp_main_pred_interwin[i]], min(main_sync_signal), max(main_sync_signal), color = 'g', linestyle = 'dashed')
        axes[1].plot(main_sync_signal[bp_main_pred_interwin[i] - 20:])
        axes[1].vlines([20], min(main_sync_signal[bp_main_pred_interwin[i] - 20:]), max(main_sync_signal[bp_main_pred_interwin[i] - 20:]), color = 'g', linestyles = 'dashed')
    else:
        axes[0].vlines([bp_main_pred_interwin[i]], min(main_sync_signal), max(main_sync_signal), color = 'g', linestyle = 'dashed')
        axes[0].vlines([bp_main_pred_interwin[i + 1]], min(main_sync_signal), max(main_sync_signal), color = 'g', linestyle = 'dashed')
        if bp_main_pred_interwin[i] - 20 < 0:
            front_sub = 0
        else:
            front_sub = 20
        if bp_main_pred_interwin[i+1] + 20 > len(bp_main_pred_interwin):
            end_add = 0
        else:
            end_add = 20
        sig = main_sync_signal[bp_main_pred_interwin[i] - front_sub:bp_main_pred_interwin[i+1] + end_add]
        axes[1].plot(sig)
        axes[1].vlines([front_sub, len(main_sync_signal[bp_main_pred_interwin[i] - front_sub:bp_main_pred_interwin[i+1]]) - end_add], min(sig), max(sig), color = 'g', linestyles = 'dashed')
    
    if plot_title:
        title = plot_title 
    else:
        title = f"Start Pred Marker {i}"
    plt.suptitle(title, fontsize = 10)

    if display:
        plt.show()

    if save_path:
        plt.savefig(save_path)
    
    plt.close()
    

def detect_interwin_frames(img_seq, fps, plot_title = None, display = False, save_path = None):
    
    # get all sync cell signals
    all_sync_signals = []
    for r in range(config.max_cells_H):
        for c in range(config.max_cells_W): 
            if not valid_r_c(r, c, config.max_cells_W, config.max_cells_H, config.localization_N) or f"{r}-{c}" in config.reserved_localization_cells:
                continue
            if (r == 0 or r == config.max_cells_H - 1 or c == 0 or c == config.max_cells_W - 1): #this a border cell, and not in reserved localization cells, so must be a channel pilot
                cell_signal = get_nonloc_cell_signal_from_imgseq(img_seq, r, c)
                all_sync_signals.append(cell_signal)

    # set signal processing parameters
    config.interwin_upper_env_rollingmax_n = 9 
    config.interwin_upper_env_rollingavg_n = 9
   
    # use average of all separate sync cell signals as one main sync  signal
    all_sync_signals = np.array(all_sync_signals)
    main_sync_signal = np.mean(np.array(all_sync_signals), axis = 0)

    # bandpass filter
    bp_main_sync_signal = main_sync_signal - main_sync_signal.mean()
    bp_main_sync_signal = butter_bandpass_filter(bp_main_sync_signal, config.frequency - config.interwin_detection_bandpass_tolerance, config.frequency + config.interwin_detection_bandpass_tolerance, fps, order=5)
    bp_main_sync_signal = pd.Series(bp_main_sync_signal)

    # get upper envelope
    rolling_max = bp_main_sync_signal.rolling(config.interwin_upper_env_rollingmax_n, center = True).max().tolist()
    for i in range(int((config.interwin_upper_env_rollingmax_n - 1)/2)):
        rolling_max[i] = rolling_max[int((config.interwin_upper_env_rollingmax_n - 1)/2)]
    for i in range(int((config.interwin_upper_env_rollingmax_n - 1)/2)):
        rolling_max[-(i+1)] = rolling_max[-(int((config.interwin_upper_env_rollingmax_n - 1)/2) + 1)]
    pos_env = pd.Series(rolling_max).rolling(config.interwin_upper_env_rollingavg_n, center = True).mean().tolist()
    left_fill = np.mean(bp_main_sync_signal[:int((config.interwin_upper_env_rollingavg_n - 1)/2)])
    right_fill = np.mean(bp_main_sync_signal[-int((config.interwin_upper_env_rollingavg_n - 1)/2):])
    for i in range(int((config.interwin_upper_env_rollingavg_n - 1)/2)):
        pos_env[i] = left_fill
    for i in range(int((config.interwin_upper_env_rollingavg_n - 1)/2)):
        pos_env[-(i+1)] = right_fill
    pos_env = np.array(pos_env)

    # use troughs in upper envelope of bandpassed loc signal as indicators of an interwindow period
    bp_main_pred_interwin, _ = find_peaks(-1 * pos_env, distance = config.min_interwin_time * fps)

    if plot_title:
        fig, axes = plt.subplots(2, tight_layout=True, figsize=(12, 6))
        plt.suptitle(f"{plot_title}\nAveraged Sync Cell")
        axes[0].set_title("Raw Signal")
        axes[0].plot(main_sync_signal)
        axes[1].set_title(f"Bandpassed w/ Envelope (rolling max n = {config.interwin_upper_env_rollingmax_n}, rolling avg n = {config.interwin_upper_env_rollingavg_n})")
        axes[1].plot(bp_main_sync_signal)
        axes[1].vlines(bp_main_pred_interwin, min(bp_main_sync_signal), max(bp_main_sync_signal), color = 'r', linestyle = 'dashed')
        axes[1].plot(pos_env, color = "purple", alpha = 0.5)

        if display:
            plt.show()

        if save_path:
            fig.savefig(save_path)
        
        plt.close()
    
        for i in range(len(bp_main_pred_interwin)):
            vis_portion(i, main_sync_signal, bp_main_pred_interwin, display = display)
        
    return bp_main_pred_interwin, main_sync_signal
  
def extract_payload_components(payload):
    enc_digest = payload[:config.digest_size]
    tag = payload[config.digest_size:config.digest_size + config.tag_size]
    return enc_digest, tag

def extract_digest_components(digest):
    bin_seq_num = digest[:config.bin_seq_num_size]

    id_feat_hash_half = digest[config.bin_seq_num_size:config.bin_seq_num_size + config.identity_hash_k // 2]

    dynamic_feat_hash = digest[config.bin_seq_num_size + config.identity_hash_k // 2:config.bin_seq_num_size + config.identity_hash_k // 2 + config.dynamic_hash_k] 

    unit_id = digest[config.bin_seq_num_size + config.identity_hash_k // 2 + config.dynamic_hash_k:config.bin_seq_num_size + config.identity_hash_k // 2 + config.dynamic_hash_k + config.unit_id_size]
    date_ordinal = digest[-config.date_ordinal_size:]

    return digest, bin_seq_num, id_feat_hash_half, dynamic_feat_hash, unit_id, date_ordinal


def get_digest_from_payload(payload):
    """
    Given payload, extract digest and return it, along with decision whether it passed the checksum, i.e., HMAC
    """
    digest, tag = extract_payload_components(payload)

    h = hmac.new(config.aes_key, digest, hashlib.sha1)
    comp_tag = h.digest()[:config.tag_size]
    comp_tag_bits = bytes_to_bitstring(comp_tag)
    if comp_tag_bits != tag:
        pass_checksum = False
    else:
        pass_checksum = True

    digest, bin_seq_num, id_feat_hash_half, dynamic_feat_hash, unit_id, date_ordinal = extract_digest_components(digest)
    
    return digest, bin_seq_num, id_feat_hash_half, dynamic_feat_hash, pass_checksum

    
def generate_localization_reference_corners(display = False):
    ref_img = np.zeros((config.slm_H, config.slm_W)).astype(np.float32)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)

    if config.localization_N is not None:
        ref_corners = []
        for i in range(4):
            c = get_loc_marker_center(i, config.slm_W, config.slm_H, config.N, config.buffer_space, config.localization_N)
            ref_corners.append(c)
            cv2.circle(ref_img, c, 2, (0, 0, 255), -1) 

        if display:
            cv2.imshow("Localization reference", ref_img)
            cv2.waitKey(0)
        return ref_img, ref_corners
    else:
        print("Unsupported localization cell type!")


def localize(video_path, heatmap_settings, force_calc_homography = False, output_path_prefix = "", manually_approve = False, display_loc = False):
    """
    Search video for localization corners and calculate homography
    """
    
    if not os.path.exists(video_path):
        print(f"ERROR: Can't find video file at {video_path}")
        return
    
    vid_name = video_path.split("/")[-1].split(".")[0]
    output_path = f"{output_path_prefix}{vid_name}"
    os.makedirs(output_path, exist_ok = True)

    heatmap_path = f"{output_path}/{vid_name}_heatmap.png"
    hom_path = f"{output_path}/{vid_name}_homography_final.pkl"

    if not os.path.exists(hom_path) or force_calc_homography:
        if os.path.exists(heatmap_path):
            heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
        else:
            start_heatmap = time.time()
            heatmap, _, _ = create_localization_heatmap(video_path,
                                                        frame_range = heatmap_settings["frame_range"],
                                                        denoise = True, display = False)
            end_heatmap = time.time()
            heatmap_time = end_heatmap - start_heatmap
            f = open(f"{output_path}/timing.txt", "a")
            if heatmap_settings["frame_range"] is not None:
                f.write(f"Heatmap extraction: {video_path},{heatmap_time},{heatmap_settings['frame_range'][1] - heatmap_settings['frame_range'][0]} frames\n")
            else:
                cap = cv2.VideoCapture(video_path)
                total_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                f.write(f"Heatmap extraction: {video_path},{heatmap_time},all {total_num_frames} frames\n")
            f.close()

            cv2.imwrite(heatmap_path, heatmap)
        

        corner_detection_start = time.time()
        corner_centers, corner_bboxes = detect_heatmap_cells(heatmap, density_diameter = heatmap_settings["density_diameter"], density_threshold  = heatmap_settings["density_threshold"], otsu_inc = heatmap_settings["otsu_inc"],  erode = heatmap_settings["erode"], area_threshold = heatmap_settings["area_threshold"], blurthensharp = heatmap_settings["blurthensharp"], kernel_dim = heatmap_settings["kernel_dim"], min_squareness = heatmap_settings["min_squareness"], display = display_loc)

        
        #get source corners to use for homography
        inferred_corner_stuff = order_calibration_code_corners(corner_centers, heatmap, slope_epsilon = heatmap_settings["slope_epsilon"], display = False)

        corner_detection_end = time.time()
        corner_detection_time = corner_detection_end - corner_detection_start
        f = open(f"{output_path}/timing.txt", "a")
        f.write(f"Corner detection: {corner_detection_time}\n")
        f.close()

        if manually_approve:
            if inferred_corner_stuff is not None:
                sorted_corner_centers, labeled_heatmap = inferred_corner_stuff
                cv2.imshow("Inferred calibration corners", labeled_heatmap)
                cv2.waitKey(0)
                accept_inferred_corners = input("Are the inferred calibration corners ok? y/n")
            if inferred_corner_stuff is None or accept_inferred_corners == "n":
                vis_corners = heatmap.copy()
                vis_corners = cv2.cvtColor(vis_corners, cv2.COLOR_GRAY2BGR)
                for c in corner_centers:
                    cv2.circle(vis_corners, c, 2, (0, 0, 255), -1)

                sorted_corner_centers = get_user_points(vis_corners) #corners MUST be in order topleft, topright, bottom left, bottom right
                if len(sorted_corner_centers) == 0:
                    sys.exit()
        else:
            return None
                    
        #perform homography between heatmap and a reference for visualization
        try:
            homography_start = time.time()
            _, reference_corner_centers = generate_localization_reference_corners(display = False)
            sample_frame = create_sample_frame()
            Hom = get_homography(sorted_corner_centers, reference_corner_centers, heatmap, sample_frame, display = display_loc)
            homography_end = time.time()
            homography_time = homography_end - homography_start
            f = open(f"{output_path}/timing.txt", "a")
            f.write(f"Homography: {homography_time},w/ display: {display_loc}\n")
            f.close()
        except Exception as e:
            print(f"Error computing homography: {e}")
            return None

        if Hom is None:
            print("Invalid homography (cv2.findHomography returned None).")
            return None
    else:
        print("LOADING HOMOGRAPHY")
        Hom = pickle.load(open(hom_path, "rb"))
    
    return Hom
    

def recover_digests(video_name, img_seq, fps, main_sync_signal, pred_interwin_markers, trial_materials_path, force_rec_digests = False,  output_path_prefix = "", display_sequences = False):
    
    vid_name = video_name.split("/")[-1].split(".")[0]
    output_path = f"{output_path_prefix}{vid_name}"
   
    rec_digests_path = f"{output_path_prefix}{vid_name}/recovered_digest_components_final.pkl"
  
    if os.path.exists(rec_digests_path) and not force_rec_digests:
        with open(rec_digests_path, "rb") as pkfile:
            digest_components = pickle.load(pkfile)
            markers_by_seq = pickle.load(pkfile)
    else:
        
        start_interwin_pred_marker = 0
        start_seq = 0


        # decode each window
        start_digest_recovery = time.time()
        curr_marker_index = start_interwin_pred_marker
        seq_num = start_seq
        last_decodable_seq = False
        curr_id_hash = None
        last_full_id_hash = None
        digest_components = []
        markers_by_seq = {}
    
        while not last_decodable_seq:
            # try:
            if curr_marker_index == len(pred_interwin_markers) - 1:
                this_img_seq = img_seq[pred_interwin_markers[curr_marker_index]:, : , :, :]
                last_decodable_seq = True
                markers_by_seq[seq_num] = [pred_interwin_markers[curr_marker_index], len(img_seq)]
            else:
                this_img_seq = img_seq[pred_interwin_markers[curr_marker_index]:pred_interwin_markers[curr_marker_index + 1], : , :, :]
                markers_by_seq[seq_num] = [pred_interwin_markers[curr_marker_index], pred_interwin_markers[curr_marker_index + 1]]
        
            gts = None
                
            tot_hard_pred, tot_probs, tot_gt, _ ,  _, _ = decode_sequence(this_img_seq, fps, display = display_sequences, gts = gts)
            
            # error correct
            correctable_payload = True
            try:
                pred_payload , pred_previterbi_encoded, correctable_payload = config.error_corrector.decode_payload(tot_probs[:config.viterbi_payload_size])
            except Exception as e:
                print(f"Unrecognizable error recovering encountered Seq {seq_num}. Reported error: {e}.")
                correctable_payload = False

            if correctable_payload:
                digest, bin_seq_num, id_feat_hash_half, dynamic_feat_hash, pass_checksum = get_digest_from_payload(pred_payload)
                rec_seq_num  = int(bin_seq_num, 2)

                if rec_seq_num % 2 == 0:
                    curr_id_hash = id_feat_hash_half
                else:
                    if curr_id_hash is not None: 
                        curr_id_hash += id_feat_hash_half
                        if len(curr_id_hash) == config.identity_hash_k:
                            # only update last full id hash when we have the full hash. If we don't, we keep the previous one.
                            # this protects against case where previous half was not recovered due to corruption
                            last_full_id_hash = curr_id_hash 
            
                if pass_checksum:
                    digest_components.append([1, rec_seq_num, last_full_id_hash, dynamic_feat_hash])
                    #print(Fore.GREEN + f"Recovered digest from encountered Seq {seq_num} (actual Seq num: {rec_seq_num})." + Style.RESET_ALL)
                else:
                    digest_components.append([0, rec_seq_num, last_full_id_hash, dynamic_feat_hash])
                    #print(Fore.RED + f"Failed to recover digest from encountered Seq {seq_num}  because of checksum failure." + Style.RESET_ALL)
            else:
                digest_components.append([0, None, None, None])
                #print(Fore.RED + f"Failed to recover digest for encountered Seq {seq_num} because of error correction failure." + Style.RESET_ALL)

            curr_marker_index += 1
            seq_num += 1

        end_digest_recovery = time.time()
        digest_recovery_time = end_digest_recovery - start_digest_recovery
        f = open(f"{output_path}/timing.txt", "a")
        f.write(f"Digest recovery: {digest_recovery_time},{len(digest_components)} windows\n")
        f.close()

    return digest_components, markers_by_seq

def get_interwin_frames(video_name, video_path, img_seq, fps, Hom, invis = False, force_pred_interwin = False, extension = ".mp4", display =  False,  output_path_prefix = ""):
    
    vid_name = video_name.split("/")[-1].split(".")[0]
    output_path = f"{output_path_prefix}{vid_name}"

    pred_interwin_markers_path = f"{output_path}/pred_interwin_markers_final.pkl"
    if not os.path.exists(pred_interwin_markers_path) or force_pred_interwin:
        start_win_pred = time.time()

        # predict window starts/end frames
        pred_interwin_markers, main_sync_signal = detect_interwin_frames(img_seq, fps, display = display, plot_title = "Interwin Preds", save_path = f"{output_path_prefix}{vid_name}/interwins.png") 
        end_win_pred = time.time()
        win_pred_time = end_win_pred - start_win_pred
        f = open(f"{output_path}/timing.txt", "a")
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_num_frames / frame_rate
        cap.release()
        f.write(f"Window prediction: {win_pred_time},{duration} second video\n")
        f.close()
        
    else:
        with open(pred_interwin_markers_path, "rb") as f:
            pred_interwin_markers = pickle.load(f)
            main_sync_signal = pickle.load(f)
    
    return pred_interwin_markers, main_sync_signal

# remove this function after done evaluation
def get_any_id_hash(digest_components):
    """
    Given digest components, return the first valid id hash encountered
    """
    for digest in digest_components:
        print(digest)
        if digest[0] == 1 and digest[2] is not None and digest[2] != -1:
            return digest[2]
    return None

# remove markers_by_seq and num_frames parameters once done evaluation
def is_valid_digest_set(digest_components, id_feats_required = True, markers_by_seq = None, num_frames = None):
    """
    Given digest components, return whether the set can be used for verification. 
    If identity features are required, this requires there being  at least one pair of consecutive valid digests, 
    with the first being an even window. Otherwise, there just needs to be one valid digest

    Parameters:
        digest_components (list): 
            list of lists of the form [validity, seq_num, id_hash, dynamic_hash], one for each window
    
    Returns:
        num_valid_digests (int): number of valid digests
    """
    if id_feats_required:
        for i, digest in enumerate(digest_components):
            if markers_by_seq is not None and num_frames is not None:
                start_frame, end_frame = markers_by_seq[i]
                print(i, digest_components[i][0], digest_components[i][1], end_frame, num_frames)
                if end_frame > num_frames:
                    continue
            else:
                print(i, digest_components[i][0], digest_components[i][1])
         
            if i == 0:
                continue
            if digest[0] == 1 and digest[2] is not None:
                return True
    else:
        # same as above but no need to check that id_hash is not None
        for i, digest in enumerate(digest_components):
            if markers_by_seq is not None and num_frames is not None:
                start_frame, end_frame = markers_by_seq[i]
                print(i, digest_components[i][0], digest_components[i][1], end_frame, num_frames)
                if end_frame > num_frames:
                    continue
            else:
                print(i, digest_components[i][0], digest_components[i][1])
         
            if i == 0:
                continue
            if digest[0] == 1:
                return True
            
        return False

    
def verify(video_name, heatmap_settings, manually_approve_corners = False, display_loc = False,
            trial_materials_path = None,
            id_feats_required = True,
            output_path_prefix = "", 
            force_calc_homography = False, force_pred_interwin = False, force_rec_digests = False,
            display_interwin = False, display_sequences = False, invis = True, extension = ".mp4"):
    """
    Verify a video stored at video_name+extension

    Parameters:
        video_name (str): Name of video
        heatmap_settings (dict): Settings for localization heatmap
        manually_approve_corners (bool): Whether to manually approve localization corners
        display_loc (bool): Whether to display localization
    """
    

    video_path = video_name + extension
    print(Fore.BLUE + f"------------------------------- VERIFYING {video_path} -------------------------------" + Style.RESET_ALL)
        
    if not os.path.exists(video_path):
        print(Fore.RED + f"Can't find video file at {video_path}" + Style.RESET_ALL)
        return

    try:
        print("Opening video...")
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        if frame_rate == 0:
            print("ERROR: Corrupt video.") # this is also a reliable way to check if the video is corrupt, as it should never occur for a valid video
            return
    except:
        print(f"ERROR: Can't open video file at {video_path}")
        return


    vid_name = video_path.split("/")[-1].split(".")[0]
    output_path = f"{output_path_prefix}{vid_name}"
     
    best_hom = None
    best_pred_interwin_markers = None
    best_main_sync_signal = None
    best_digest_components = None
    best_markers_by_seq = None
    selected_hom_num = None
    hom_num = 0
    final_hom_selected = False
    hom_path = f"{output_path}/{vid_name}_homography_final.pkl"
    if os.path.exists(hom_path):
        print("HOM EXISTS")
        final_hom_selected = True
    for otsu_num, otsu_inc in enumerate(heatmap_settings["otsu_inc"]):
        for area_num, area_threshold in enumerate(heatmap_settings["area_threshold"]):
            for min_squareness_num, min_squareness in enumerate(heatmap_settings["min_squareness"]):
                print(f"Obtaining homography with otsu inc = {otsu_inc} ({otsu_num + 1}/{len(heatmap_settings['otsu_inc'])}), area threshold = {area_threshold} ({area_num}/{len(heatmap_settings['area_threshold'])}), min squareness = {min_squareness} ({min_squareness_num}/{len(heatmap_settings['min_squareness'])}).")
               
                # localize
                print("Getting homography...")
                this_heatmap_settings = heatmap_settings.copy()
                this_heatmap_settings["otsu_inc"] = otsu_inc
                this_heatmap_settings["area_threshold"] = area_threshold
                this_heatmap_settings["min_squareness"] = min_squareness
                Hom = localize(video_path, this_heatmap_settings, force_calc_homography = force_calc_homography, output_path_prefix = output_path_prefix, display_loc = display_loc, manually_approve = manually_approve_corners)
                if Hom is None:
                    continue

                # load video
                print("Loading video...")
                start_load = time.time()
                img_seq, fps = loadVideo(video_path, colorspace = config.colorspace, Hom = Hom)
                end_load = time.time()
                f = open(f"{output_path}/timing.txt", "a")
                f.write(f"Load video: {end_load - start_load}\n")
                f.close()

                # get interwin predictions
                print("Predicting interwindow frames...")
                pred_interwin_markers, main_sync_signal = get_interwin_frames(vid_name, video_path, img_seq, fps, Hom, invis = invis, force_pred_interwin = force_pred_interwin, extension = extension, display = display_interwin, output_path_prefix = output_path_prefix)

                # recover digests
                print("Recovering digests...")
                digest_components, markers_by_seq = recover_digests(video_name, img_seq, fps, main_sync_signal, pred_interwin_markers, trial_materials_path, force_rec_digests = force_rec_digests, output_path_prefix = output_path_prefix, display_sequences = display_sequences)
                
                # below num_frames and markers_by_seq things only relevant for copied digests, remove after experiments done
                cap = cv2.VideoCapture(video_path)
                total_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                validity = is_valid_digest_set(digest_components, id_feats_required = id_feats_required)
                if validity:
                    print(Fore.GREEN + "Valid digest set" + Style.RESET_ALL)
                else:
                    print(Fore.RED + "Invalid digest set" + Style.RESET_ALL)
          
                if validity: # don't bother with others, this homography was solid
                    best_hom = Hom
                    best_digest_components = digest_components
                    best_markers_by_seq = markers_by_seq
                    best_pred_interwin_markers = pred_interwin_markers
                    best_main_sync_signal = main_sync_signal
                    selected_hom_num = hom_num
                    selected_hom_num = hom_num
                    final_hom_selected = True
                    break

                if final_hom_selected:
                    break
                
                hom_num += 1

            if final_hom_selected:
                break

        if final_hom_selected:
            break

    if best_hom is None:
        print(Fore.RED + "Failed to localize" + Style.RESET_ALL)
        f = open(f"{output_path}/outcome.csv", "w")
        f.write("Failed in localization")
        f.close()      
        return 
    
    # save the heatmap settings and chosen ones, only if the homography was generated during this run, not just loaded from existing
    if not os.path.exists(hom_path):
        with open(f"{output_path}/heatmap_settings.pkl", "wb") as f:
            pickle.dump(heatmap_settings, f)
            pickle.dump(this_heatmap_settings, f)

    f = open(f"{output_path}/outcome.csv", "w")
    f.write(f"Selected hom: {selected_hom_num}\n")
    f.close()
    
    # save the data corresponding to the best Hom as the final ones
    with open(f"{output_path}/{vid_name}_homography_final.pkl", "wb") as pklfile:
        pickle.dump(best_hom, pklfile)
    
    with open(f"{output_path}/pred_interwin_markers_final.pkl", "wb") as f:
        pickle.dump(best_pred_interwin_markers, f)
        pickle.dump(best_main_sync_signal, f)

    with open(f"{output_path_prefix}{vid_name}/recovered_digest_components_final.pkl", "wb") as pkfile:
        pickle.dump(best_digest_components, pkfile)
        pickle.dump(best_markers_by_seq, pkfile)
 

    #initialize video digest extractor
    print("Verifying all sequences...")
    print("Initializing video digest extractor...")
    vid_digest_extractor = VideoDigestExtractor(video_path)
    print("Done initalizing video digest extractor.")

    verifiable_seqs = sorted(list(markers_by_seq.keys()))[:-1]
   
    id_dists = []
    dyn_dists = []
    num_verified_seqs = 0
    first_time_id = True # remove after experiments done
    for i, ver_seq_num in enumerate(verifiable_seqs):
        
        print(Fore.MAGENTA + f"----------- VERIFYING ENCOUNTERED VERIFIABLE SEQ {ver_seq_num} -----------" + Style.RESET_ALL)
        reference_digest = digest_components[i +1] # the digest embedded in the next window, whose contents we will use to verify this window's video
    
        # if i + 1 >= len(digest_components): # what is this for??
        #     print("Embedded digest cut off in video. Skipping verification")
        #     print("------------------------------------------------")
        #     continue

        if reference_digest[0] == 0:
            print("Corrupt sequence digest. Skipping verification")
            print("------------------------------------------------")
            continue
        
        rec_seq_num = reference_digest[1]
        rec_id_hash = reference_digest[2]
        rec_dynamic_hash = reference_digest[3]
        print(f"Using digest from Seq: {rec_seq_num}.")

       
        
        start_frame, end_frame = markers_by_seq[ver_seq_num]

        if end_frame - start_frame < config.video_window_duration * fps * 0.9: # add some tolerance with 0.9, otherwise valid windows are discounted
            print(f"Window too short (should be at least {config.video_window_duration * fps * 0.9} frames, is {end_frame - start_frame} frames long). Skipping.")
            continue
   
        # below only relevant for copied digests, remove after experiments done
        if end_frame > total_num_frames:
            print("Window extends beyond video. Skipping.")
            continue
        
        # below for experiments only
        # for the very first iteration, set the rec_id_hash to any valid ID hash from the copied digest components
        if first_time_id == True:
            print(Fore.YELLOW + "ID hash not recovered. Using any valid ID hash from copied digest components." + Style.RESET_ALL)
            rec_id_hash = get_any_id_hash(digest_components)
            print(f"Using ID hash: {rec_id_hash}")
            first_time_id = False

        if rec_id_hash is None:
            print("ID hash not recovered.")
            id_dists.append(-1)
        else:
            start_id_feat =  time.time()
            ver_id_hash, ver_id_vec = vid_digest_extractor.get_id_features_hash(start_frame)
            print("Ver ID hash: ", ver_id_hash)
            end_id_feat = time.time()
            if ver_id_hash is None: #additional check to ensure that a None is never hashed
                id_hash_dist = -1
            else:
                id_hash_dist = hamming(ver_id_hash, rec_id_hash)
            print(Fore.MAGENTA + f"ID hash dist: {id_hash_dist}" + Style.RESET_ALL)
            f = open(f"{output_path}/timing.txt", "a")
            f.write(f"ID feature hash extraction for win {ver_seq_num}: {end_id_feat - start_id_feat}\n")
            f.close()
            id_dists.append(id_hash_dist)

    
        if end_frame - start_frame > config.video_window_duration * fps * 1.05: # don't use excessively large frames
            start_frame = end_frame - int(config.video_window_duration * fps * 1.05)

        start_dyn_feat = time.time()
        # TODO: direct comparison
        ver_dynamic_hash_dist, ver_optimal_start_frame = get_dynamic_hash_dist(video_path, start_frame, end_frame, rec_dynamic_hash, verification_dir = f"{output_path_prefix}{vid_name}")
        end_dyn_feat = time.time()
        dyn_feat_time = end_dyn_feat - start_dyn_feat
        f = open(f"{output_path}/timing.txt", "a")
        f.write(f"Dynamic feature hash extraction for win {ver_seq_num}: {dyn_feat_time}\n")
        f.close()
        dyn_dists.append(ver_dynamic_hash_dist)
        print(Fore.MAGENTA + f"Min dynamic hash dist: {ver_dynamic_hash_dist}. Best shift: {ver_optimal_start_frame - start_frame}" + Style.RESET_ALL)
        print("------------------------------------------------")

        num_verified_seqs += 1

    print(Fore.BLUE + "--------------------- FINAL --------------------- ")
    if num_verified_seqs == 0:
        print("No sequences were verified.")
        f = open(f"{output_path}/outcome.csv", "w")
        f.write("Failed post-localization")
        f.close()
    else:  
        max_id = max(id_dists)
        max_dyn = max(dyn_dists)
        f = open(f"{output_path}/outcome.csv", "a")
        out_id_dists = ";".join([str(d) for d in id_dists])
        out_dyn_dists = ";".join([str(d) for d in dyn_dists])
        f.write(f"{video_path},{out_id_dists},{out_dyn_dists},{max_id},{max_dyn}\n")
        f.close()

        print(f"Max ID dist: {max_id}.\nMax dyn dist: {max_dyn}")

    print("------------------------------------------------" + Style.RESET_ALL)


otsu_incs = [10, -30, -50, 0, -20, 20, -10, 30, -40, 40]

heatmap_settings = {
    "erode": 1,
    "kernel_dim": 5,
    "blurthensharp": False,
    "area_threshold": [[1000, 25000], [200, 25000], [100, 50000]],
    "min_squareness": [0.6, 0.8, None],
    "otsu_inc": otsu_incs,
    "density_diameter": 200,
    "density_threshold": 280,
    "frame_range": [400, 800],#None,
    "slope_epsilon" : 0.1
}
  
video_name = "/media/lex/E380-1E91/Deepfake/End_To_End/deepfakes_may24/colman/talklip/googlepixel/p4_df"
verify(video_name, heatmap_settings, manually_approve_corners = True, display_loc = True,
        trial_materials_path = None,
        id_feats_required = True,
        output_path_prefix = "./",
        force_calc_homography = False, force_pred_interwin = False, force_rec_digests = False,
        display_interwin = True, display_sequences = True, invis = False, extension = ".mp4")