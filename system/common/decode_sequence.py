"""
Decode a window of videos, i.e., raw extract its signature data
"""

from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.pyplot as plt
from colorama import Style, Fore, Back
import sys

import config
from decoding_utils import valid_r_c, get_nonloc_cell_signal_from_imgseq, get_normalized_mag_at_freq

sys.path.append('../embedding/')
from perceptibility_utils import get_on_off_cell_val_from_cell_signal, CIEDE2000
from decoding_utils import maxSum

def decode_sequence(image_sequence, fps, get_perceptibilities = False, gts = None, pkl_path_prefix = None, signal_pkls_path = None, display = False):
    """
    Parameters:
        image_sequence: list of np.array
            list of video frames consituting the window, homographied already
        fps: int
            framerate of this video, FPS
        get_perceptibilities: if True, will return perceptibility scores for each cell
        gts: list of ground truth bits for each cell
        pkl_path_prefix: if provided, will save the signals for each cell to a pkl file with this prefix
        signal_pkls_path: if provided, will load the signals for each cell from a pkl file with this prefix
        display: if True, will display plots of the signals and decoding process
    """
    
    num_bits_per_cell = config.frequency * config.embedding_window_duration
    
    # get sync cell signals 
    pilot_mags = []
    pilot_blinks_by_pilot = []
    pilot_signals = []
    for r in range(config.max_cells_H):
        for c in range(config.max_cells_W): 
            if not valid_r_c(r, c, config.max_cells_W, config.max_cells_H, config.localization_N) or f"{r}-{c}" in config.reserved_localization_cells:
                continue
            
            if (r == 0 or r == config.max_cells_H - 1 or c == 0 or c == config.max_cells_W - 1): # this a border cell, and not in reserved localization cells, so must be a sync cell
                
                if not signal_pkls_path:
                    cell_signal = get_nonloc_cell_signal_from_imgseq(image_sequence, r, c)
                else: #load this pilot from the pkl at the path instead of extracting from img sequence
                    with open(signal_pkls_path + f"_sync_r{r}_c{c}.pkl", "rb") as pklfile:
                        cell_signal = pickle.load(pklfile)

                norm_power = get_normalized_mag_at_freq(cell_signal, fps)
                pilot_mags.append(norm_power)

                if pkl_path_prefix: # for saving for later analysis, if pkl_path_prefix is provided
                    with open(pkl_path_prefix + f"_sync_r{r}_c{c}.pkl", "wb") as pklfile:
                        pickle.dump(cell_signal, pklfile)   

                smoothed_cell_signal, fully_processed_cell_signal,  _ = process_signal(cell_signal, rollingavg_n = config.signal_rollingavg_n, detrending_rollingavg_n = config.signal_detrending_rollingavg_n, detrending_rollingmin_n = config.signal_detrending_rollingmin_n) 
                pilot_signals.append(cell_signal)
                pilot_blinks = get_individ_sync_cell_blinks(fully_processed_cell_signal, fps)
                if pilot_blinks is not None:
                    pilot_blinks_by_pilot.append(pilot_blinks)
                

                if display:
                    fig, axes = plt.subplots(3)
                    axes[0].set_title("Raw Cell Signal")
                    axes[0].plot(cell_signal)
                    axes[1].set_title("Smoothed Cell Signal")   
                    axes[1].plot(smoothed_cell_signal)
                    axes[2].set_title("Fully Processed Cell Signal")
                    axes[2].plot(fully_processed_cell_signal)
                    axes[2].vlines(pilot_blinks, min(fully_processed_cell_signal), max(fully_processed_cell_signal), color = 'r', linestyle = 'dashed')
                    plt.suptitle(f"Sync Cell Cell {r}-{c}")
                    plt.show()

    # get blinks based on blinks of averaged sync cells
    pilot_signals = np.array(pilot_signals)
    avgd_pilot_signal = np.mean(np.array(pilot_signals), axis = 0)
    smoothed_avgd_pilot_signal, fully_processed_avgd_pilot_signal, _ = process_signal(avgd_pilot_signal, rollingavg_n = config.signal_rollingavg_n, detrending_rollingavg_n = config.signal_detrending_rollingavg_n, detrending_rollingmin_n = config.signal_detrending_rollingmin_n)
    sync_blinks = get_overall_sync_blinks(fully_processed_avgd_pilot_signal, fps)
    if display:
        fig, axes = plt.subplots(2)
        plt.suptitle("Avgd Sync Signal and Final Sync Signal Blinks")
        axes[0].plot(avgd_pilot_signal)
        axes[1].plot(fully_processed_avgd_pilot_signal)
        axes[1].vlines(sync_blinks, min(fully_processed_avgd_pilot_signal), max(fully_processed_avgd_pilot_signal), color = 'r', linestyle = 'dashed')
        plt.show()
    
    # decode non-channel or localization cells
    # optionally, get perceptibility score of each
    tot_probs = [] # probs for all cells
    tot_hard_pred = ""
    tot_gt = ""
    info_cell_num = 0
    all_perceptibilities = [] #perceptibilities for all cells. will be filled if get_perceptibilities = True
    all_off_cell_bgrs = []
          
    for r in range(config.max_cells_H):
        for c in range(config.max_cells_W): 
            if not valid_r_c(r, c, config.max_cells_W,  config.max_cells_H, config.localization_N): #skip any non-border cells covered by a localization corner, if in use
                continue
            
            #this not a border cell or covered by localization corner, so decode it to recover info 
            if not  (r == 0 or r == config.max_cells_H - 1 or c == 0 or c == config.max_cells_W - 1): 
                if not signal_pkls_path:
                    cell_signal = get_nonloc_cell_signal_from_imgseq(image_sequence, r, c)
                else:
                    with open(signal_pkls_path + f"_info_r{r}_c{c}.pkl", "rb") as pklfile:
                        cell_signal = pickle.load(pklfile)

                if gts is not None:
                    gt = gts[info_cell_num]
                    if len(gt) != num_bits_per_cell:
                        print(Fore.YELLOW + Back.RED + f"WARNING: Cell {r}-{c} gt is too short ({len(gt)} bits intead of {num_bits_per_cell}). Terminating decoding." + Style.RESET_ALL)
                        break
                else:
                    gt = None

                if pkl_path_prefix:
                    with open(pkl_path_prefix + f"_info_r{r}_c{c}.pkl", "wb") as pklfile:
                        pickle.dump(cell_signal, pklfile)
                        pickle.dump(gt, pklfile)

                # if no blinks recovered, assume all 0s
                if sync_blinks is not None:
                    if display:
                        cell_hard_pred, cell_probs =  decode_data_cell_signal(sync_blinks, cell_signal, gt = gt, plot_title = f"Cell {r}-{c}")
                    else:
                        cell_hard_pred, cell_probs =  decode_data_cell_signal(sync_blinks, cell_signal)
                else:
                    cell_hard_pred = ''.join(['0' for i in range(int(config.embedding_window_duration * config.frequency))])
                    cell_probs = [0 for i in range(int(config.embedding_window_duration * config.frequency))]

                tot_probs += cell_probs

                if gt is not None:
                    tot_gt += gt
                tot_hard_pred += cell_hard_pred
                info_cell_num += 1
            
            if get_perceptibilities:
                # get perceptibility + BGR for all cells, not just info ones
                cell_top = r * (config.N + config.buffer_space) 
                cell_left = c * (config.N + config.buffer_space) 
                cell_bottom = cell_top + config.N
                cell_right = cell_left + config.N

                off_cell_bgr, on_cell_bgr = get_on_off_cell_val_from_cell_signal(cell_signal, image_sequence, [cell_top, cell_left, cell_bottom, cell_right])
                perceptibility = CIEDE2000(off_cell_bgr[::-1], on_cell_bgr[::-1])
                all_perceptibilities.append(perceptibility)
                all_off_cell_bgrs.append(off_cell_bgr)
    
    return tot_hard_pred, tot_probs, tot_gt, pilot_mags, all_perceptibilities, all_off_cell_bgrs



def process_signal(raw_cell_signal, rollingavg_n = 2, detrending_rollingmin_n = None, detrending_rollingavg_n = None, fps = None):
    smoothed_cell_signal = pd.Series(raw_cell_signal).rolling(rollingavg_n, center=True).mean().tolist()
    for i in range(int((rollingavg_n- 1)/2) + 1):
        smoothed_cell_signal[i] = smoothed_cell_signal[int((rollingavg_n- 1)/2) + 1]
    for i in range(int((rollingavg_n - 1)/2)):
        smoothed_cell_signal[-(i+1)] = smoothed_cell_signal[-(int((rollingavg_n- 1)/2) + 1)]

    fully_processed_cell_signal = None
    neg_env = None
    if detrending_rollingmin_n is not None:
        rolling_min = pd.Series(smoothed_cell_signal).rolling(detrending_rollingmin_n, center = True).min().tolist()
        for i in range(int((detrending_rollingmin_n - 1)/2)):
            if i >= len(rolling_min):
                break
            rolling_min[i] = rolling_min[min(int((detrending_rollingmin_n- 1)/2), len(rolling_min) - 1)]
        for i in range(int((detrending_rollingmin_n - 1)/2)):
            if i < 0:
                break
            rolling_min[-(i+1)] = rolling_min[max(int((detrending_rollingmin_n- 1)/2) * -1 + 1, 0)]

        neg_env = pd.Series(rolling_min).rolling(detrending_rollingavg_n, center=True).mean().tolist()
        left_fill = np.mean(smoothed_cell_signal[:int((detrending_rollingavg_n - 1)/2)])
        right_fill = np.mean(smoothed_cell_signal[-int((detrending_rollingavg_n - 1)/2):])
        for i in range(int((detrending_rollingavg_n - 1)/2)):
            if i >= len(neg_env):
                break
            neg_env[i] = left_fill
        for i in range(int((detrending_rollingavg_n - 1)/2)):
            if i < 0:
                break
            neg_env[-(i+1)] = right_fill
        smoothed_cell_signal = np.array(smoothed_cell_signal)
        neg_env = np.array(neg_env)
        fully_processed_cell_signal = smoothed_cell_signal - neg_env
        fully_processed_cell_signal = fully_processed_cell_signal.tolist()
    else:
        fully_processed_cell_signal = smoothed_cell_signal
        
    return smoothed_cell_signal, fully_processed_cell_signal, neg_env


def get_overall_sync_blinks(pilot_signal, fps):
    """
    Given a pilot signal, recorded at fps <fps> return the indices corresponding 
    to frames in the signal at which a blink (either cell on or cell off) occurred
    """

    pilot_signal = np.array(pilot_signal)
  
    min_dist = int((1/(config.frequency))*fps) * .4
    min_prominence = 0.05 #0.1
    pilot_peaks, pilot_peak_properties  = find_peaks(pilot_signal, distance = min_dist, prominence = min_prominence, plateau_size=(None,None), height=(None, None), width = (None, None))
    pilot_troughs, pilot_trough_properties = find_peaks(-1*pilot_signal, distance = min_dist, prominence = min_prominence, plateau_size=(None,None), height=(None, None), width = (None, None))

    if len(pilot_peaks) == 0:
        #return dummy template
        template = [i* int((1/(config.frequency) / 2)*fps) for i in range(int(config.embedding_window_duration * config.frequency) * 2)]
        template = np.array(template)
        template += 3 #start???
        return template

    # filter peaks, considering only those in the largest continuous subsequence of detected peaks by prominence
    pilot_peak_prominences = pilot_peak_properties['prominences']
    _, target_pilot_peak_is = maxSum(pilot_peak_prominences, len(pilot_peak_prominences), int(config.embedding_window_duration * config.frequency))

    # if short peaks, assess whether they should be at start or end and add as many as necessary
    num_missing_peaks = int(config.embedding_window_duration * config.frequency) - len(target_pilot_peak_is)
    if num_missing_peaks > 0:
        num_added_peaks = 0
        added_peaks = []
        step = int((1/(config.frequency))*fps) # interpeak/intertrough distance
        if pilot_peaks[target_pilot_peak_is][0] < 10: # add to beginning if there appears to be space. Use 10 as a heuristic here.
            # add equally spaces peaks to start until correct number of peaks present or no space to add
            next_front_peak = pilot_peaks[target_pilot_trough_is][0] - step #initialize
            while num_added_peaks < num_missing_peaks and next_front_peak > 0:
                added_peaks.append(next_front_peak)
                next_front_peak -= step
                num_added_peaks += 1
        next_end_peak = pilot_peaks[target_pilot_peak_is][-1] + step # otherwise add to end
        while num_added_peaks < num_missing_peaks and next_end_peak < len(pilot_signal):
            # add equally spaces peaks to end until correct number of peaks present or no space to add
            added_peaks.append(next_end_peak)
            next_end_peak -= step
            num_added_peaks += 1
        # in worst case, there are simply not enough peaks and the above attempts to reconcile this fail, 
        # so just tack on from end one after another so decoding can proceed
        if num_added_peaks < num_missing_peaks:
            next_end_peak = len(pilot_signal) - 1
            while num_added_peaks < num_missing_peaks:
                added_peaks.append(next_end_peak)
                next_end_peak -= 1
                num_added_peaks += 1
        added_peaks = np.array(added_peaks)
        target_pilot_peaks = np.concatenate((pilot_peaks[target_pilot_peak_is], added_peaks))
    else:
        # too many/just the right amount of peaks detected
        target_pilot_peaks = pilot_peaks[target_pilot_peak_is]

    # filter troughs, only considering those occuring bewteen peaks
    target_pilot_trough_is = []
    for i in target_pilot_peak_is:
        for j, t in enumerate(pilot_troughs):
            if i < len(pilot_peaks) - 1:
                if t > pilot_peaks[i] and t < pilot_peaks[i + 1]:
                    target_pilot_trough_is.append(j)
                    break 
    
    # if short or over troughs, assess whether they should be at start or end and add as many as necessary
    num_missing_troughs = int(config.embedding_window_duration * config.frequency) - len(target_pilot_trough_is)
    if num_missing_troughs == 1:
        # most common case - just last trough missing. Using last peak as reference has worked well in practice.
        target_pilot_troughs = np.concatenate((pilot_troughs[target_pilot_trough_is], np.array([target_pilot_peaks[-1] + int((1/(config.frequency))*fps/2)])))
    elif len(target_pilot_trough_is) != int(config.embedding_window_duration * config.frequency):
        # more than one trough missing, we will add multiple. Start by adding to the end with space from the last trough
        added_troughs = []
        step = int((1/(config.frequency))*fps) # interpeak/intertrough distance
        next_end_trough = target_pilot_peaks[-1] + step
        num_added_troughs = 0
        while num_added_troughs < num_missing_troughs and next_end_trough < len(pilot_signal):
            added_troughs.append(next_end_trough)
            next_end_trough += step
            num_added_troughs += 1
        # in worst case, there are simply not enough troughs and the above attempts to reconcile this fail, 
        # so just tack on from end one after another so decoding can proceed
        if num_added_troughs < num_missing_troughs:
            next_end_trough = len(pilot_signal) - 1
            while num_added_troughs < num_missing_troughs:
                added_troughs.append(next_end_trough)
                next_end_trough -= 1
                num_added_troughs += 1
        added_troughs = np.array(added_troughs)
        target_pilot_troughs = np.concatenate((pilot_troughs[target_pilot_trough_is], added_troughs))
        # target_pilot_troughs = np.concatenate((pilot_troughs[target_pilot_trough_is], np.array([target_pilot_peaks[-1] + int((1/(config.frequency))*fps/2)])))
    else:
        # too many/just the right amount of troughs detected
        target_pilot_troughs = pilot_troughs[target_pilot_trough_is]

    target_sync_cell_blinks = np.concatenate((target_pilot_peaks, target_pilot_troughs)) 
    
    # sort all the blinks
    target_sync_cell_blinks = np.sort(target_sync_cell_blinks)
    return target_sync_cell_blinks


def get_individ_sync_cell_blinks(pilot_signal, fps):
    """
    Given a pilot signal, recorded at fps <fps> return the indices corresponding 
    to frames in the signal at which a blink (either cell on or cell off) occurred
    If there aren't the expected number of peaks and troughs (with an exception if it 
    is just short one peak, one trough, which we assume is due to the first peak or 
    last trough not being detected and can adjust for), return None
    """

    pilot_signal = np.array(pilot_signal)
  
    min_dist = int((1/(config.frequency))*fps) * .4
    min_prominence = 0.1
    pilot_peaks, pilot_peak_properties  = find_peaks(pilot_signal, distance = min_dist, prominence = min_prominence, plateau_size=(None,None), height=(None, None), width = (None, None))
    pilot_troughs, pilot_trough_properties = find_peaks(-1*pilot_signal, distance = min_dist, prominence = min_prominence, plateau_size=(None,None), height=(None, None), width = (None, None))

    # filter peaks, considering only those in the largest continuous subsequence of detected peaks by prominence
    pilot_peak_prominences = pilot_peak_properties['prominences']
    _, target_pilot_peak_is = maxSum(pilot_peak_prominences, len(pilot_peak_prominences), int(config.embedding_window_duration * config.frequency))
        
    # filter troughs, only considering those occuring bewteen peaks
    target_pilot_trough_is = []
    for i in target_pilot_peak_is:
        for j, t in enumerate(pilot_troughs):
            if i < len(pilot_peaks) - 1:
                if t > pilot_peaks[i] and t < pilot_peaks[i + 1]:
                    target_pilot_trough_is.append(j)
                    break
  
    target_pilot_peaks = pilot_peaks[target_pilot_peak_is]
    
    # if short a peak, add one at the start, defined as the first considered peak - a period
    if len(target_pilot_peak_is) == int(config.embedding_window_duration * config.frequency)- 1:
        target_pilot_peaks = np.concatenate((np.array([pilot_peaks[target_pilot_peak_is][0]]) - int((1/(config.frequency))*fps), pilot_peaks[target_pilot_peak_is]))
    else:
        target_pilot_peaks = pilot_peaks[target_pilot_peak_is]

    # if short a trough, add one at the end, defined as last considered peak + half a period
    if len(target_pilot_trough_is) == int(config.embedding_window_duration * config.frequency )- 1:
        #target_pilot_troughs = np.concatenate((pilot_troughs[target_pilot_trough_is], np.array([pilot_peak_properties['right_bases'][target_pilot_peak_is[-1]]])))
        target_pilot_troughs = np.concatenate((pilot_troughs[target_pilot_trough_is], np.array([target_pilot_peaks[-1] + int((1/(config.frequency))*fps/2)])))
    else:
        target_pilot_troughs = pilot_troughs[target_pilot_trough_is]
    
    if len(target_pilot_peaks) != int(config.embedding_window_duration * config.frequency)  or len(target_pilot_troughs) != int(config.embedding_window_duration * config.frequency):
        return None

    target_sync_cell_blinks = np.concatenate((target_pilot_peaks, target_pilot_troughs)) 
    
    # sort all the blinks
    target_sync_cell_blinks = np.sort(target_sync_cell_blinks)

    return target_sync_cell_blinks


def decode_data_cell_signal(sync_cell_blinks, raw_cell_signal, plot_title = None,  fps = None, d = 0, gt = None, extra_vlines = None):
    """
    Decode a data cell signal using one set of sync_cell_blinks. Return both the hard decisions (0 or 1) and soft decisions (float representing nearness to 1 or 0).
    plot_title: Title for plot, if desired. If no plot_title is provided, a plot will not be produced.
    gt: optional ground truth of the cell signal, useful for adaptation or debugging.
    detrend_n: Size of window ucsed to calculate lower envelop of the cell signal, used for detrending it. If none, detrending
    will not be performed
    """
    smoothed_cell_signal, fully_processed_cell_signal, neg_env = process_signal(raw_cell_signal, rollingavg_n = config.signal_rollingavg_n, detrending_rollingmin_n = config.signal_detrending_rollingmin_n, detrending_rollingavg_n = config.signal_detrending_rollingavg_n, fps = fps)

    pred_string = ['0' for i in range(int(config.frequency * config.embedding_window_duration))]
    probs = [0 for i in range(int(config.frequency * config.embedding_window_duration))]
    diffs = []
    for i in range(0, len(sync_cell_blinks), 2):
        if i + 1 >= len(sync_cell_blinks):
            break
        t1 = int(sync_cell_blinks[i])
        t2 = int(sync_cell_blinks[i + 1])
        p1_avg = np.mean(fully_processed_cell_signal[t1:t1+d+1])
        p2_avg = np.mean(fully_processed_cell_signal[t2:t2+d+1])
        diffs.append(p1_avg - p2_avg)
        if p1_avg < p2_avg:
            pred_string[int(i/2)] = '1'
        probs[int(i/2)] = p1_avg - p2_avg
    pred_string = "".join(pred_string)

    if plot_title is not None:
        ymin = min(fully_processed_cell_signal)
        ymax = max(fully_processed_cell_signal)

       
        if config.signal_detrending_rollingmin_n:
            fig, axes = plt.subplots(2, tight_layout = True)
            if gt is not None:
                title = plot_title + f"\nGT:    {gt}"
            else:
                title = plot_title
            axes[0].plot(smoothed_cell_signal)
            axes[0].plot(neg_env, alpha=0.6)
            axes[1].plot(fully_processed_cell_signal)
            axes[1].vlines(sync_cell_blinks, ymin, ymax, linestyles = "dashed", colors = "grey")
            if extra_vlines is not None:
                axes[1].vlines(extra_vlines, ymin, ymax, linestyles = "dashed", colors = "red", alpha=0.5)
            for i in range(0, len(sync_cell_blinks), 2):
                axes[1].text(sync_cell_blinks[i], ymin + (ymax - ymin)/3, f"{probs[int(i/2)]:.2f}")
                if gt is not None:
                    if pred_string[int(i/2)] != gt[int(i/2)]:
                        axes[1].text(sync_cell_blinks[i], ymin + (ymax - ymin)/2, pred_string[int(i/2)], c = 'r')
                        axes[0].set_title(f"{title}\nPred: {pred_string}", fontdict ={'color':'red','size':10})
                    else:
                        axes[0].set_title(f"{title}\nPred: {pred_string}", fontsize = 10)
                else:
                    axes[1].text(sync_cell_blinks[i], ymin + (ymax - ymin)/2, pred_string[int(i/2)])   
                    axes[0].set_title(f"{title}\nPred: {pred_string}", fontsize = 10)
            plt.show()
        else:
            if gt is not None:
                title = plot_title + f"\nGT:   {gt}"
            else:
                title = plot_title
            plt.title(f"{title}\nPred: {pred_string}", fontsize = 10)
            plt.plot(smoothed_cell_signal)
            plt.vlines(sync_cell_blinks, ymin, ymax, linestyles = "dashed", colors = "grey")
            if extra_vlines is not None:
                plt.vlines(extra_vlines, ymin, ymax, linestyles = "dashed", colors = "red", alpha=0.5)
            for i in range(0, len(sync_cell_blinks), 2):
                plt.text(sync_cell_blinks[i], ymin + (ymax - ymin)/3, f"{probs[int(i/2)]:.2f}")
                if gt is not None:
                    if pred_string[int(i/2)] != gt[int(i/2)]:
                        plt.text(sync_cell_blinks[i], ymin + (ymax - ymin)/2, pred_string[int(i/2)], c = 'r') 
                else:
                    plt.text(sync_cell_blinks[i], ymin + (ymax - ymin)/2, pred_string[int(i/2)])
                    plt.title(f"{title}\nPred: {pred_string}", fontsize = 10)
            plt.show()
            plt.clf()
            
    return pred_string, probs

