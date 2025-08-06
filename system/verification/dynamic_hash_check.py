
'''
Given a dynamic hash extracted from the digest, and a video, this script scans through videos starting/end at small shift from the predicted window start marker 
to find the shift that leads to the smallest dynamic hash distance to the input dynamic hash,
and returns this distance and said shift
'''
import sys
import os
import cv2
import numpy as np
import math
import pickle
import time
import warnings
from sklearn.exceptions import  InconsistentVersionWarning
warnings.filterwarnings(action='ignore', category =  InconsistentVersionWarning)
warnings.filterwarnings(action='ignore', category = FutureWarning)
sys.path.append("../common")
from digest_extraction import MPExtractor, create_dynamic_hash_from_dynamic_features
from rp_lsh import hamming
import config

def save_video_signals(video_path, output_pkl_path):
    """
    Originally in /deepfake_detection/system/evaluation/dynamic_features/extract_signals.py
    Copied here so everything needed for verification is in one place
    """

    mp_extractor = MPExtractor()

    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    dynamic_features = []
    pose = []
    face = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_feats, face_bbox, detection_result = mp_extractor.extract_features(frame, config.target_features)
        dynamic_features.append(frame_feats)
        try:
            pose.append(detection_result.facial_transformation_matrixes)
        except Exception as err:
            # at some harsh cam angles, transformation matrix cannot be extracted
            pose.append(np.nan)
        
        try:
            face.append(face_bbox)
        except Exception as err:
            face.append(np.nan)

        frame_num += 1

    with open(output_pkl_path, "wb") as pklfile:
        pickle.dump(dynamic_features, pklfile)
        pickle.dump(pose, pklfile)
        pickle.dump(face, pklfile)


def get_dynamic_hash_dist(video_path, win_start_frame, win_end_frame, input_digest, verification_dir):

    epsilon = config.window_alignment_epsilon

    # extract signals for the whole video, if not already done
    signal_pkl_path = f"{verification_dir}/video_signals.pkl"
    if not os.path.exists(signal_pkl_path):
        start_time = time.time()
        save_video_signals(video_path, signal_pkl_path)
        end_time = time.time()
        f = open(f"{verification_dir}/timing.txt", "a")
        f.write(f"Mediapipe signal extraction (included in first dynamic hash time): {end_time - start_time}\n")
        f.close()
    
    # scan through target windows of the video to find a best match
    with open(signal_pkl_path, "rb") as pklfile:
        dynamic_features = pickle.load(pklfile)
        pose = pickle.load(pklfile)
        face_bbox = pickle.load(pklfile)

    dynamic_fam = config.dynamic_fam
    dynamic_hash_funcs = config.dynamic_hash_funcs 

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    scan_start = win_start_frame - epsilon
    #scan_end = win_end_frame + epsilon - config.video_window_duration*fps
    scan_end = win_start_frame + epsilon

    min_dyn_feat_dist = config.dynamic_hash_k
    best_start = 0
    print("window start, end: ", win_start_frame, win_end_frame)
    print("scan_start, scan_end: ", scan_start, scan_end)
    for i in range(scan_start, int(scan_end+1)):
        if int(i + config.video_window_duration*fps+1) >= len(dynamic_features):
            break
        cur_dynamic_features = dynamic_features[i:int(i + config.video_window_duration*fps+1)]
        cur_dynamic_hash, _, _, _  = create_dynamic_hash_from_dynamic_features(cur_dynamic_features, dynamic_fam, dynamic_hash_funcs)
        cur_dynamic_hash_dist =  hamming(input_digest, cur_dynamic_hash)
        if cur_dynamic_hash_dist < min_dyn_feat_dist:
            min_dyn_feat_dist = cur_dynamic_hash_dist
            best_start = i

    return min_dyn_feat_dist, best_start




