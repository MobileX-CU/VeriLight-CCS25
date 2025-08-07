"""
Utils for decoding pixel-level signals to extract signatures
"""
import cv2
import numpy as np
import sys
from scipy.signal import butter, filtfilt
sys.path.append('../common')
import config

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    use filtfilt instead of lfilter to avoid any shifting of signal
    https://dsp.stackexchange.com/questions/19084/applying-filter-in-scipy-signal-use-lfilter-or-filtfilt
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def loadVideo(video_path, colorspace = 'bgr', downsamples = 0, crop_coords = None, Hom = None):
    """
    Given a video path, load it as an array of images with the dimensions
    [num_frames, img_H, img_W, 3]
    From following Eulerian Video Magnification implementation found online: 
    https://github.com/hbenbel/Eulerian-Video-Magnification/

    Parameters:
        video_path: str
            path to video file
        colorspace : str 
            colorspace to convert to. Options are 'bgr', 'yuv', 'ycrcb'
        downsamples : int
            number of times to downsample the video
        crop_coords : list
            coordinates to crop the video to. Format is x1, y1, x2, y2
    """
    if crop_coords is not None and downsamples > 0:
        print("CAREFUL, you are both cropping and downsampling. Do you really want to do this?")

    image_sequence = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    while video.isOpened():
        ret, frame = video.read()
        if ret is False:
            break
       
        if crop_coords is not None: 
            x1, y1, x2, y2 = crop_coords
            frame = frame[y1:y2, x1:x2, :]
        
        if downsamples > 0:
            for i in range(downsamples):
                frame = cv2.pyrDown(frame)
        
        if colorspace == 'yuv':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        elif colorspace == 'ycrcb':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

        if Hom is not None:
            frame = cv2.warpPerspective(frame, Hom, (640, 360))

        # cv2.imshow("loadVideo", frame)
        # if cv2.waitKey(25) & 0xFF == ord('q'): 
        #     break
  
        image_sequence.append(frame[:, :, :])

    video.release()

    return np.asarray(image_sequence), fps


def get_homography(sorted_contour_centers, reference_centers, heatmap, reference_img, display = True):
    """
    Given contour centers from heatmap and reference centers from a priori encoding parameters,
    calculate homography necessary to transform source defined by contour centers to destination of references
    """
    sorted_contour_centers = np.array(sorted_contour_centers)
    reference_centers = np.array(reference_centers)
    H, status = cv2.findHomography(sorted_contour_centers, reference_centers)

  
    # Warp source image to destination based on homography to visualize success
    if display:
        if len(heatmap.shape) < 3:
            vis_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        else:
            vis_heatmap = heatmap
        img_out = cv2.warpPerspective(vis_heatmap, H, (config.slm_W, config.slm_H))
        cv2.imshow("Src image", vis_heatmap)
        cv2.imshow("Dst image", reference_img)
        cv2.imshow("Result", img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # weird workadound to make destroyAllWindows work
        # https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv
        for i in range (1,5):
            cv2.waitKey(1)     
    return H


def get_normalized_mag_at_freq(signal, fps):
    """
    Get power of signal at localization frequency and normalize it by the mean power in the
    frequency bands specified as lower range, upper range, etc. in config
    """
    signal = signal - np.mean(signal)
    ps =  np.abs(np.fft.fft(signal))**2
    freqs = np.fft.fftfreq(len(signal), 1/fps)

    # get target freq power
    lower_freq_index = (np.abs(freqs - config.localization_frequency)).argmin()
    upper_freq_index = None
    for i, f in enumerate(freqs):
        if i < lower_freq_index and np.abs(config.localization_frequency - f) < config.target_lower_epsilon:
            lower_freq_index = i
        if i > lower_freq_index and np.abs(config.localization_frequency - f) < config.target_upper_epsilon:
            upper_freq_index = i + 1 #add 1 bc slicing would cut here
    if upper_freq_index is None:
        upper_freq_index = lower_freq_index + 1
    target_freq_power = np.sum(ps[lower_freq_index:upper_freq_index])
    
    # get "noise" power
    lower_range_start_index = (np.abs(freqs - config.lower_range_start)).argmin()
    lower_range_end_index = (np.abs(freqs - config.lower_range_end)).argmin()
    upper_range_start_index = (np.abs(freqs - config.upper_range_start)).argmin()
    upper_range_end_index = (np.abs(freqs - config.upper_range_end)).argmin()
    normalization_power =  np.mean(ps[[i for i in range(lower_range_start_index, lower_range_end_index)] + [i for i in range(upper_range_start_index, upper_range_end_index)]])
    
    return target_freq_power / normalization_power


def set_dec_params(override_slm_W = None, override_slm_H = None, 
               override_N = None, override_buffer_space = None, 
               override_localization_N = None, override_target_channel = None, override_analysis_colorspace = None,
               override_frequency = None, override_localization_frequency = None,
               override_embedding_window_duration = None,
               override_lower_range_start = None, override_lower_range_end = None, 
                override_upper_range_start  = None, override_upper_range_end = None, 
                override_target_lower_epsilon = None, override_target_upper_epsilon = None, ):
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
    
    if override_localization_frequency is not None:
        localization_frequency = override_localization_frequency
    else:
        localization_frequency = config.localization_frequency
    
    if override_embedding_window_duration is not None:
        embedding_window_duration = override_embedding_window_duration
    else:
        embedding_window_duration = config.embedding_window_duration

    max_cells_W = int((slm_W ) / (N + buffer_space))
    max_cells_H = int((slm_H + buffer_space) / (N + buffer_space))

        
    if localization_N is not None:
        max_cells = max_cells_H * max_cells_W - 4*localization_N**2
        num_pilot_cells = max_cells_W * 2 + max_cells_H * 2 - 4 - ((2*localization_N - 1)*4)
        reserved_localization_cells = []
    else:
        max_cells = max_cells_H * max_cells_W 
        num_pilot_cells = max_cells_W * 2 + max_cells_H * 2 - 4
        reserved_localization_cells = [] #"r-c" index format. leave empty if don't want any reserved.
        count = 0
        for c in range(max_cells_W):
            if count % 2 == 0:
                reserved_localization_cells.append(f"{0}-{c}")
            count += 1
        for r in range(1, max_cells_H):
            if count % 2 == 0:
                reserved_localization_cells.append(f"{r}-{max_cells_W-1}")
            count += 1
        for c in range(max_cells_W-2, -1, -1):
            if count % 2 == 0:
                reserved_localization_cells.append(f"{max_cells_H-1}-{c}")
            count += 1
        for r in range(max_cells_H-2, 0, -1):
            if count % 2 == 0:
                reserved_localization_cells.append(f"{r}-0")
            count += 1

      
    if override_lower_range_start is not None:
        lower_range_start = override_lower_range_start
    else:
        lower_range_start = config.lower_range_start
    
    if override_lower_range_end is not None:
        lower_range_end = override_lower_range_end
    else:
        lower_range_end = config.lower_range_end
    
    if override_upper_range_start is not None:
        upper_range_start = override_upper_range_start
    else:
        upper_range_start = config.upper_range_start
    
    if override_upper_range_end is not None:
        upper_range_end = override_upper_range_end
    else:
        upper_range_end = config.upper_range_end
    
    if override_target_lower_epsilon is not None:
        target_lower_epsilon = override_target_lower_epsilon
    else:
        target_lower_epsilon = config.target_lower_epsilon
    
    if override_target_upper_epsilon is not None:
        target_upper_epsilon = override_target_upper_epsilon
    else:
        target_upper_epsilon = config.target_upper_epsilon

    if override_target_channel is not None:
        target_channel = override_target_channel
    else:
        target_channel = config.target_channel
    
    if override_analysis_colorspace is not None:
        analysis_colorspace = override_analysis_colorspace
    else:
        analysis_colorspace = config.colorspace
    

    return slm_W, slm_H, N, buffer_space, localization_N, max_cells_W, max_cells_H, reserved_localization_cells, target_channel, analysis_colorspace, frequency, localization_frequency, embedding_window_duration, max_cells, lower_range_start, lower_range_end, upper_range_start, upper_range_end, target_lower_epsilon, target_upper_epsilon


def valid_r_c(r, c, max_cells_W,  max_cells_H, localization_N):
    """
    If localization corners being used, determine if cell at index specified by r, c 
    is a valid pilot or info cell, or if it would be at a position occupied by a localization marker
    """

    if localization_N is None:
        return True
    if (r < localization_N and c < localization_N) or \
    (r < localization_N and c >= max_cells_W - localization_N) or \
    (r >= max_cells_H - localization_N and  c < localization_N) or\
    (r >= max_cells_H - localization_N and c >= max_cells_W - localization_N):
        return False
    else:
        return True

def get_loc_marker_center(loc_marker_num, slm_W, slm_H, N, buffer_space, localization_N):

    max_cells_W = int((slm_W ) / (N + buffer_space))
    max_cells_H = int((slm_H + buffer_space) / (N + buffer_space))

    if loc_marker_num == 0:
        cell_top = 0
        cell_left = 0
    elif loc_marker_num == 1:
        cell_top = 0
        cell_left =  (max_cells_W - localization_N) * (N + buffer_space)
    elif loc_marker_num == 2:
        cell_top = (max_cells_H - localization_N) * (N + buffer_space) 
        cell_left = 0
    else:
        cell_top = (max_cells_H - localization_N) * (N + buffer_space)
        cell_left = (max_cells_W - localization_N) * (N + buffer_space)

    #loc_cell_dim =  (localization_N * N) + (localization_N - 1) * buffer_space - buffer_space
    loc_cell_dim = localization_N * (N + buffer_space) - buffer_space
    return cell_left + loc_cell_dim//2, cell_top + loc_cell_dim//2


def get_localization_signal_from_imgseq(image_sequence, loc_marker_num, target_channel, slm_W, slm_H, N, buffer_space, localization_N, max_cells_W, max_cells_H, padding = 0, display = False):
    """
    get signal of localization marker. also return the cell boundaries because they are useful
    in next step of adaptation (getting BGRs)
    """
  
    if loc_marker_num == 0:
        cell_top = padding
        cell_left = padding
    elif loc_marker_num == 1:
        cell_top = padding
        cell_left =  (max_cells_W - localization_N) * (N + buffer_space) + padding
    elif loc_marker_num == 2:
        cell_top = (max_cells_H - localization_N) * (N + buffer_space) + padding
        cell_left = padding
    else:
        cell_top = (max_cells_H - localization_N) * (N + buffer_space) + padding
        cell_left = (max_cells_W - localization_N) * (N + buffer_space) + padding
 
    # loc_cell_dim =  (localization_N * N) + (localization_N - 1) * buffer_space
    loc_cell_dim = localization_N * (N + buffer_space) - buffer_space
    cell_bottom = cell_top + loc_cell_dim - padding
    cell_right = cell_left + loc_cell_dim - padding

    mask = np.zeros((slm_H, slm_W), np.uint8)
    cv2.rectangle(mask, (cell_left, cell_top), (cell_right, cell_bottom), 255, -1)   
    
    signal = []
    for i in range(image_sequence.shape[0]):
        img = image_sequence[i, :,:, :]
        if target_channel == "sum":
            mean_cell_brightness = cv2.mean(img, mask=mask)[0]
            mean_cell_brightness += cv2.mean(img, mask=mask)[1]
            mean_cell_brightness += cv2.mean(img, mask=mask)[2]
        else:
            mean_cell_brightness = cv2.mean(img, mask=mask)[target_channel]
        
        if display:
            cv2.rectangle(img, (cell_left, cell_top), (cell_right, cell_bottom), 255, 1)   
            cv2.imshow("localization marker", img)
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break

        signal.append(mean_cell_brightness)
        
    return signal, [cell_top, cell_left, cell_bottom, cell_right]


def get_nonloc_cell_signal_from_imgseq(image_sequence, cell_row, cell_col, display = False):
    cell_top = cell_row * (config.N + config.buffer_space) 
    cell_left = cell_col * (config.N + config.buffer_space) 
    cell_bottom = cell_top + config.N
    cell_right = cell_left + config.N

    mask = np.zeros((config.slm_H, config.slm_W), np.uint8)
    cv2.rectangle(mask, (cell_left, cell_top), (cell_right, cell_bottom), 255, -1)   

    signal = []
    for i in range(image_sequence.shape[0]):
        img = image_sequence[i, :,:, :]
        if display:
            cv2.rectangle(img, (cell_left, cell_top), (cell_right, cell_bottom), 255, 1)
            cv2.imshow(f"Cell {i}", img)
            cv2.waitKey(0)
        if config.target_channel == "sum":
            mean_cell_brightness = cv2.mean(img, mask = mask)[0]
            mean_cell_brightness += cv2.mean(img, mask = mask)[1]
            mean_cell_brightness += cv2.mean(img, mask = mask)[2]
        else:
            mean_cell_brightness = cv2.mean(img, mask=mask)[config.target_channel]
        signal.append(mean_cell_brightness)
        
    return signal


def maxSum(arr, n, k):
    """"
    modified solution found at https://www.geeksforgeeks.org/find-maximum-minimum-sum-subarray-size-k/
    """
    # k must be smaller than n
    if (n < k):
        return -1, [i for i in range(len(arr))]
    
    indices = [i for i in range(k)]
     
    # Compute sum of first
    # window of size k
    res = 0
    for i in range(k):
        res += arr[i]
 
    # Compute sums of remaining windows by
    # removing first element of previous
    # window and adding last element of 
    # current window.
    curr_sum = res
    for i in range(k, n):
        curr_sum += arr[i] - arr[i-k]
        if res < curr_sum:
            indices = [r for r in range(i-k+1, i+1)]
            res = curr_sum
    return res, indices