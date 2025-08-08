"""
Verify a video
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import cv2
from scipy.signal import find_peaks
from colorama import Fore, Style
import sys
import hmac
import hashlib

from create_heatmap import create_localization_heatmap
from dynamic_hash_check import get_dynamic_hash_dist
from visualize import visualize_features


sys.path.append('../common/')
import config
from digest_extraction import VideoFeatureExtractor
from decoding_utils import loadVideo, get_homography, get_loc_marker_center, valid_r_c, get_nonloc_cell_signal_from_imgseq, butter_bandpass_filter
from decode_sequence import decode_sequence
from bitstring_utils import bytes_to_bitstring, bitstring_to_bytes
from rp_lsh import hamming, hash_point

sys.path.append('../embedding/')
from calibration_utils import  get_user_points, detect_heatmap_cells, order_calibration_code_corners
from psk_encode_minimal import create_sample_frame

ID_THRESH = 42
DYN_THRESH = 56

def detect_interwin_frames(img_seq, fps, plot_title = None, display = False, save_path = None):
    
    # get all sync cell signals
    all_sync_signals = []
    for r in range(config.max_cells_H):
        for c in range(config.max_cells_W): 
            if not valid_r_c(r, c, config.max_cells_W, config.max_cells_H, config.localization_N) or f"{r}-{c}" in config.reserved_localization_cells:
                continue
            if (r == 0 or r == config.max_cells_H - 1 or c == 0 or c == config.max_cells_W - 1): # this a border cell, and not in reserved localization cells, so must be a sync cell
                cell_signal = get_nonloc_cell_signal_from_imgseq(img_seq, r, c)
                all_sync_signals.append(cell_signal)
        
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
        print("Plotting interwindow detection results...")
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

        # detailed per-window sync signal visualization, uncomment if desired for debugging
        # for i in range(len(bp_main_pred_interwin)):
        #     vis_window_sync_signal(i, main_sync_signal, bp_main_pred_interwin, display = display)
        
    return bp_main_pred_interwin, main_sync_signal


def get_digest_from_payload(payload):
    """
    Given payload, extract digest and return it, along with decision whether it passed the checksum, i.e., HMAC
    """
    digest = payload[:config.digest_size]
    tag = payload[config.digest_size:config.digest_size + config.tag_size * 8]
  
    digest_bytes = bitstring_to_bytes(digest)
    h = hmac.new(config.key, digest_bytes, hashlib.sha1)
    comp_tag = h.digest()[:config.tag_size]
    comp_tag_bits = bytes_to_bitstring(comp_tag)

    if comp_tag_bits != tag:
        pass_checksum = False
    else:
        pass_checksum = True

    bin_seq_num = digest[:config.bin_seq_num_size]

    id_feat_hash_half = digest[config.bin_seq_num_size:config.bin_seq_num_size + config.identity_hash_k // 2]
    # if id_feat_hash_half is all zeros, that means the ID feature couldn't be extracted on core unit during embedding.
    # this is the convention established in digest_extraction.py's create_digest_from_features() 
    if id_feat_hash_half.count("0") == config.identity_hash_k // 2:
        id_feat_hash_half = None

    dynamic_feat_hash = digest[config.bin_seq_num_size + config.identity_hash_k // 2:config.bin_seq_num_size + config.identity_hash_k // 2 + config.dynamic_hash_k] 

    unit_id = digest[config.bin_seq_num_size + config.identity_hash_k // 2 + config.dynamic_hash_k:config.bin_seq_num_size + config.identity_hash_k // 2 + config.dynamic_hash_k + config.unit_id_size]
    date_ordinal = digest[-config.date_ordinal_size:]
   
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


def localize(video_path, heatmap_settings, output_path = "", force_calc_homography = False, manually_approve = False, display_loc = False):
    """
    Search video for localization corners and calculate homography
    """
    
    if not os.path.exists(video_path):
        print(f"ERROR: Can't find video file at {video_path}")
        return
        
    heatmap_path = f"{output_path}/heatmap.png"
    hom_path = f"{output_path}/homography.pkl"
  
    if not os.path.exists(hom_path) or force_calc_homography:
        if os.path.exists(heatmap_path):
            heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
        else:
            heatmap, _, _ = create_localization_heatmap(video_path,
                                                        frame_range = heatmap_settings["frame_range"],
                                                        denoise = True, display = False)
            cv2.imwrite(heatmap_path, heatmap)
        
        corner_centers, corner_bboxes = detect_heatmap_cells(heatmap, density_diameter = heatmap_settings["density_diameter"], density_threshold  = heatmap_settings["density_threshold"], otsu_inc = heatmap_settings["otsu_inc"],  erode = heatmap_settings["erode"], area_threshold = heatmap_settings["area_threshold"], blurthensharp = heatmap_settings["blurthensharp"], kernel_dim = heatmap_settings["kernel_dim"], min_squareness = heatmap_settings["min_squareness"], display = display_loc)
        
        #get source corners to use for homography
        inferred_corner_stuff = order_calibration_code_corners(corner_centers, heatmap, slope_epsilon = heatmap_settings["slope_epsilon"], display = False)

   
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
                    
        # perform homography between heatmap and a reference for visualization
        try:
            _, reference_corner_centers = generate_localization_reference_corners(display = False)
            sample_frame = create_sample_frame()
            Hom = get_homography(sorted_corner_centers, reference_corner_centers, heatmap, sample_frame, display = display_loc)
           
        except Exception as e:
            print(f"Error computing homography: {e}")
            return None

        if Hom is None:
            print("Invalid homography (cv2.findHomography returned None).")
            return None

        with open(f"{output_path}/homography.pkl", "wb") as pklfile:
            pickle.dump(Hom, pklfile)
    else:
        print(f"Loading existing homography from {hom_path}...")
        Hom = pickle.load(open(hom_path, "rb"))
    
    return Hom
    

def recover_digests(img_seq, fps, pred_interwin_boundaries, force_rec_digests = False,  output_path = "", display_sequences = False):
       
    rec_digests_path = f"{output_path}/recovered_digest_components.pkl"
  
    if os.path.exists(rec_digests_path) and not force_rec_digests:
        with open(rec_digests_path, "rb") as pkfile:
            digest_components = pickle.load(pkfile)
            boundaries_by_seq = pickle.load(pkfile)
    else:
        
        # decode each window
        curr_boundary_index = 0
        seq_num = 0
        last_decodable_seq = False
        curr_id_hash = None
        last_full_id_hash = None
        digest_components = []
        boundaries_by_seq = {}
    
        while not last_decodable_seq:
            print(f"Decoding sequence {seq_num}...")

            if curr_boundary_index == len(pred_interwin_boundaries) - 1:
                this_img_seq = img_seq[pred_interwin_boundaries[curr_boundary_index]:, : , :, :]
                last_decodable_seq = True
                boundaries_by_seq[seq_num] = [pred_interwin_boundaries[curr_boundary_index], len(img_seq)]
            else:
                this_img_seq = img_seq[pred_interwin_boundaries[curr_boundary_index]:pred_interwin_boundaries[curr_boundary_index + 1], : , :, :]
                boundaries_by_seq[seq_num] = [pred_interwin_boundaries[curr_boundary_index], pred_interwin_boundaries[curr_boundary_index + 1]]
                        
            tot_hard_pred, tot_probs, _, _ ,  _, _ = decode_sequence(this_img_seq, fps, display = display_sequences)
            
            # error correct
            correctable_payload = True
            try:
                pred_payload , pred_previterbi_encoded, correctable_payload = config.error_corrector.decode_payload(tot_probs[:config.viterbi_payload_size])
                pred_payload = pred_payload[:config.payload_size]
            except Exception as e:
                print(f"Unrecognizable error recovering encountered Seq {seq_num}. Reported error: {e}.")
                correctable_payload = False

            if correctable_payload:
                digest, bin_seq_num, id_feat_hash_half, dynamic_feat_hash, pass_checksum = get_digest_from_payload(pred_payload)
                rec_seq_num  = int(bin_seq_num, 2)

                if id_feat_hash_half is not None:
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

            curr_boundary_index += 1
            seq_num += 1
              
        with open(rec_digests_path, "wb") as pkfile:
            pickle.dump(digest_components, pkfile)
            pickle.dump(boundaries_by_seq, pkfile)


    return digest_components, boundaries_by_seq

def get_interwin_frames_boundaries(img_seq, fps, output_path = "", force_pred_interwin = False,  display =  False):
        
    pred_interwin_boundaries_path = f"{output_path}/pred_interwin_boundaries.pkl"
    if not os.path.exists(pred_interwin_boundaries_path) or force_pred_interwin:
        # predict window starts/end frames
        pred_interwin_boundaries, main_sync_signal = detect_interwin_frames(img_seq, fps, display = display, plot_title = "Interwin Preds", save_path = f"{output_path}/interwins.png") 
      
        with open(f"{output_path}/pred_interwin_boundaries.pkl", "wb") as f:
            pickle.dump(pred_interwin_boundaries, f)
            pickle.dump(main_sync_signal, f)

    else:
        with open(pred_interwin_boundaries_path, "rb") as f:
            pred_interwin_boundaries = pickle.load(f)
            main_sync_signal = pickle.load(f)
    
    return pred_interwin_boundaries, main_sync_signal


def verify(video_path, heatmap_settings, manually_approve_corners = False, display_loc = False,
            output_path = "", 
            force_calc_homography = False, force_pred_interwin = False, force_rec_digests = False,
            display_interwin = False, display_sequences = False):
    """

    Parameters:
        video_path (str): Path to the video file
        heatmap_settings (dict): Settings for localization heatmap
        manually_approve_corners (bool): Whether to manually approve localization corners
        display_loc (bool): Whether to display localization
    """

    print(Fore.BLUE + f"------------------------------- VERIFYING {video_path} -------------------------------" + Style.RESET_ALL)
    
    # make sure video exists at video_path
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
    
    # create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # localize
    print("Getting homography...")
    Hom = localize(video_path, heatmap_settings, force_calc_homography = force_calc_homography, output_path = output_path, display_loc = display_loc, manually_approve = manually_approve_corners)
    if Hom is None:
        print("Couldn't obtain homography.")
        return

    # load video for interwindow prediction and digest recovery
    print("Loading video frames...")
    img_seq, fps = loadVideo(video_path, colorspace = config.colorspace, Hom = Hom)
    
    # get interwin predictions
    print("Predicting interwindow frames...")
    pred_interwin_boundaries, main_sync_signal = get_interwin_frames_boundaries(img_seq, fps, force_pred_interwin = force_pred_interwin, display = display_interwin, output_path = output_path)

    # recover digests
    print("Recovering digests...")
    digest_components, boundaries_by_seq = recover_digests(img_seq, fps, pred_interwin_boundaries, force_rec_digests = force_rec_digests, output_path = output_path, display_sequences = display_sequences)
    verifiable_seqs = sorted(list(boundaries_by_seq.keys()))[:-1] # list of window numbers that can be verified, i.e., those that have a digest embedded in the next window

    #initialize video digest extractor and extract MP features
    print("Extracting MP features...")
    vid_feature_extractor = VideoFeatureExtractor(video_path, output_path)
    dynamic_features, pose, face_bbox, raw_detection_results = vid_feature_extractor.extract_mp_features()


    print("Verifying sequences...")
    # tracking of dists/hashes/seq nums for visualization purposes
    id_dists = []
    dyn_dists = []
    id_hashes = []
    dyn_hashes = []
    rec_id_hashes = []
    rec_dynamic_hashes = []
    rec_seq_nums = []
    num_verified_seqs = 0
    for i, ver_seq_num in enumerate(verifiable_seqs):
        
        reference_digest = digest_components[i +1] # the digest embedded in the next window, whose contents we will use to verify this window's video

        if reference_digest[0] == 0:
            print("Corrupt sequence digest. Skipping verification")
            print("------------------------------------------------")
            # tracking of dists/hashes/seq nums for visualization purposes
            id_dists.append(-1)
            dyn_dists.append(-1)
            id_hashes.append("N/A. Signature unrecoverable")
            dyn_hashes.append("N/A. Signature unrecoverable")
            rec_seq_nums.append(None)
            continue
        
        rec_seq_num = reference_digest[1]
        rec_id_hash = reference_digest[2]
        rec_dynamic_hash = reference_digest[3]
        rec_seq_nums.append(rec_seq_num) # tracking for visualization purposes
        print(Fore.MAGENTA + f"------------- EMBEDDING WINDOW {rec_seq_num} -------------" + Style.RESET_ALL)
        start_frame, end_frame = boundaries_by_seq[ver_seq_num]

        if end_frame - start_frame < config.video_window_duration * fps * 0.9: # add some tolerance with 0.9, otherwise valid windows are discounted
            print(f"Window too short (should be at least {config.video_window_duration * fps * 0.9} frames, is {end_frame - start_frame} frames long). Skipping.")
             # tracking of dists/hashes/seq nums for visualization purposes
            id_dists.append(-1)
            dyn_dists.append(-1)
            rec_id_hashes.append("N/A. Incomplete video window")
            rec_dynamic_hashes.append("N/A. Incomplete video window")
            id_hashes.append("N/A. Incomplete video window")
            dyn_hashes.append("N/A. Incomplete video window")
            continue
  
        if rec_id_hash is None:
            print("ID hash not recovered.")
            rec_id_hashes.append("N/A. ID hash not recovered")
            id_dists.append(-1)
            id_hashes.append("N/A.")
        else:
            rec_id_hashes.append(rec_id_hash) # tracking for visualization purposes
            # verify every frame in the window against the recovered ID hash
            max_window_id_hash_dist = -1
            max_window_id_hash = None
            for frame_num in range(start_frame, end_frame):
                ver_id_hash, ver_id_vec = vid_feature_extractor.get_id_features_hash(frame_num)
                if ver_id_hash is None: 
                    id_hash_dist = -1
                else:
                    id_hash_dist = hamming(ver_id_hash, rec_id_hash)
                if id_hash_dist > max_window_id_hash_dist:
                    max_window_id_hash_dist = id_hash_dist
                    max_window_id_hash = ver_id_hash
            print(Fore.MAGENTA + f"Window max ID hash Hamming distance: {max_window_id_hash_dist}" + Style.RESET_ALL)
            id_dists.append(max_window_id_hash_dist)
            id_hashes.append(max_window_id_hash if max_window_id_hash is not None else "N/A. ID hash not recovered")
        
        if end_frame - start_frame > config.video_window_duration * fps * 1.05: # don't use excessively large windows
            start_frame = end_frame - int(config.video_window_duration * fps * 1.05)
        rec_dynamic_hashes.append(rec_dynamic_hash)
        dynamic_hash_dist, dynamic_hash = get_dynamic_hash_dist(video_path, start_frame, rec_dynamic_hash, dynamic_features)
        dyn_dists.append(dynamic_hash_dist)
        dyn_hashes.append(dynamic_hash)
        print(Fore.MAGENTA + f"Window dynamic hash Hamming distance: {dynamic_hash_dist}." + Style.RESET_ALL)
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

    print("Generating visualization video...")
    visualize_features(video_path, output_path, fps, boundaries_by_seq, verifiable_seqs, id_hashes, dyn_hashes, id_dists, dyn_dists, rec_seq_nums, rec_id_hashes, rec_dynamic_hashes) # only MP for now

   

heatmap_settings = {
    "erode": 1,
    "kernel_dim": 5,
    "blurthensharp": False,
    "area_threshold": [1000, 25000],
    "min_squareness": 0.6,
    "otsu_inc": 10,
    "density_diameter": 200,
    "density_threshold": 280,
    "frame_range": [400, 800],#None,
    "slope_epsilon" : 0.1
}
  
video_path = "/Users/hadleigh/verilight2.mp4"
verify(video_path, heatmap_settings, manually_approve_corners = True, display_loc = True,
        output_path = "verilight2_output",
        force_calc_homography = False, force_pred_interwin = False, force_rec_digests = False,
        display_interwin = False, display_sequences = False)


