
'''
Given a dynamic hash extracted from the digest, and a video, this script scans through videos starting/end at small shift from the predicted window start marker 
to find the shift that leads to the smallest dynamic hash distance to the input dynamic hash,
and returns this distance and said shift
'''
import sys
import cv2
import warnings
from sklearn.exceptions import  InconsistentVersionWarning
warnings.filterwarnings(action='ignore', category =  InconsistentVersionWarning)
warnings.filterwarnings(action='ignore', category = FutureWarning)

sys.path.append("../common")
from digest_extraction import create_dynamic_hash_from_dynamic_features
from rp_lsh import hamming
import config


def get_dynamic_hash_dist(video_path, win_start_frame, input_digest, dynamic_features):

    epsilon = config.window_alignment_epsilon


    # compare at minor (millisecond) offsets of predicted window boundary
    dynamic_fam = config.dynamic_fam
    dynamic_hash_funcs = config.dynamic_hash_funcs 
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    scan_start = win_start_frame - epsilon
    scan_end = win_start_frame + epsilon
    min_dyn_feat_dist = config.dynamic_hash_k
    final_raw_signals = None
    final_proc_signals = None
    final_concat_processed_signal = None
    opt_start_frame = 0
    for i in range(scan_start, int(scan_end+1)):
        if int(i + config.video_window_duration*fps+1) >= len(dynamic_features):
            break
        curr_dynamic_features = dynamic_features[i:int(i + config.video_window_duration*fps+1)]
        curr_dynamic_hash, curr_raw_signals, curr_proc_signals, curr_concat_processed_signal  = create_dynamic_hash_from_dynamic_features(curr_dynamic_features, dynamic_fam, dynamic_hash_funcs)
        curr_dynamic_hash_dist =  hamming(input_digest, curr_dynamic_hash)
        if curr_dynamic_hash_dist < min_dyn_feat_dist:
            opt_start_frame = i
            min_dyn_feat_dist = curr_dynamic_hash_dist
            final_raw_signals = curr_raw_signals
            final_proc_signals = curr_proc_signals
            final_concat_processed_signal = curr_concat_processed_signal

    return min_dyn_feat_dist, opt_start_frame, final_raw_signals, final_proc_signals, final_concat_processed_signal




