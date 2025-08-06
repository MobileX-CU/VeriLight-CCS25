"""
Create heatmap from a video
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
import sys
import os
sys.path.append('../common/')
from decoding_utils import loadVideo
sys.path.append('../common/')
import config


def generate_heatmap(images, fps, 
            target_freqs, lower_range_start, lower_range_end, upper_range_start, upper_range_end, target_lower_epsilon, target_upper_epsilon, target_channel,
            denoise = True, display = False):
    """
    Create heatmap from image sequence
    """
    print(f"FPS: {fps}. Target freqs: {target_freqs}. Lower range start: {lower_range_start}. Lower range end: {lower_range_end}. Upper range start: {upper_range_start}. Upper range end: {upper_range_end}. Target lower epsilon: {target_lower_epsilon}. Target upper epsilon: {target_upper_epsilon}. Target channel: {target_channel}. Denoise: {denoise}. Display: {display}")

    proc_images = images.copy()
    if target_channel == "sum":   
        proc_images = np.sum(proc_images, axis = 3)  

    proc_images = proc_images - np.mean(proc_images, axis = 0)   
    ps = np.abs(np.fft.fft(proc_images, axis = 0))**2
    freqs = np.fft.fftfreq(proc_images.shape[0], 1 / fps)

    if target_channel == "sum":
        if denoise:
            lower_range_start_index = (np.abs(freqs - lower_range_start)).argmin()
            lower_range_end_index = (np.abs(freqs - lower_range_end)).argmin()

            upper_range_start_index = (np.abs(freqs - upper_range_start)).argmin()
            upper_range_end_index = (np.abs(freqs -  upper_range_end)).argmin()

            noise_heatmap = np.mean(ps[[i for i in range(lower_range_start_index, lower_range_end_index)] + [i for i in range(upper_range_start_index, upper_range_end_index)],:,:], axis = 0)

            if display:
                ax = sns.heatmap(noise_heatmap, linewidth=0) 
                plt.title('Noise')
                plt.show()

        heatmap = None            
        for target_freq in target_freqs:
            lower_freq_index = (np.abs(freqs - target_freq)).argmin()
            upper_freq_index = None
            for i, f in enumerate(freqs):
                if i < lower_freq_index and np.abs(target_freq - f) < target_lower_epsilon:
                    lower_freq_index = i
                if i > lower_freq_index and np.abs(target_freq - f) < target_upper_epsilon:
                    upper_freq_index = i + 1 #add 1 bc slicing would cut here
            if upper_freq_index is None:
                upper_freq_index = lower_freq_index + 1

            print("Target freq heatmap: ", lower_freq_index, upper_freq_index)
            target_freq_heatmap = np.sum(ps[lower_freq_index:upper_freq_index,:,:], axis = 0)  #should this instead be mean???
        
            if display:
                ax = sns.heatmap(target_freq_heatmap , linewidth=0) 
                plt.title(f'Target : {target_freq} Hz')
                plt.show()

            if heatmap is None:
                heatmap = target_freq_heatmap
            else:
                heatmap += target_freq_heatmap
            
        if display:
            ax = sns.heatmap(heatmap, linewidth=0) 
            plt.title('Summed')
            plt.show()

        if denoise:
            heatmap = heatmap / noise_heatmap
            heatmap[heatmap < 0] = 0

        if display:
            ax = sns.heatmap(heatmap , linewidth=0) 
            plt.title('Final')
            plt.show()

        png_heatmap = minmax_scale(heatmap.flatten()).reshape(heatmap.shape) * 255
        png_heatmap  = png_heatmap.astype("uint8")
        
        return png_heatmap, None, None
    else:
        if denoise:
            lower_range_start_index = (np.abs(freqs - lower_range_start)).argmin()
            lower_range_end_index = (np.abs(freqs - lower_range_end)).argmin()

            upper_range_start_index = (np.abs(freqs - upper_range_start)).argmin()
            upper_range_end_index = (np.abs(freqs -  upper_range_end)).argmin()

            noise_heatmaps = np.mean(ps[[i for i in range(lower_range_start_index, lower_range_end_index)] + [i for i in range(upper_range_start_index, upper_range_end_index)],:,:,:], axis = 0)

            if display:
                ax = sns.heatmap(noise_heatmaps[:,:,0] , linewidth=0) 
                plt.title(f'Channel 1 Noise')
                plt.show()

                ax = sns.heatmap(noise_heatmaps[:,:,1] , linewidth=0) 
                plt.title(f'Channel 2 Noise')
                plt.show()

                ax = sns.heatmap(noise_heatmaps[:,:,2], linewidth=0) 
                plt.title(f'Channel 3 Noise')
                plt.show()
    

        heatmaps = None
        for target_freq in target_freqs:
            
            lower_freq_index = (np.abs(freqs - target_freq)).argmin()
            upper_freq_index = None
            for i, f in enumerate(freqs):
                if i < lower_freq_index and np.abs(target_freq - f) < target_lower_epsilon:
                    lower_freq_index = i
                if i > lower_freq_index and np.abs(target_freq - f) < target_upper_epsilon:
                    upper_freq_index = i + 1 #add 1 bc slicing would cut here
            if upper_freq_index is None:
                upper_freq_index = lower_freq_index + 1

            target_freq_heatmaps = np.sum(ps[lower_freq_index:upper_freq_index,:,:,:], axis = 0)  #should this instead be mean???
        
      
            if display:
                ax = sns.heatmap(target_freq_heatmaps[:,:,0] , linewidth=0) 
                plt.title(f'Channel 1 Target : {target_freq} Hz')
                plt.show()

                ax = sns.heatmap(target_freq_heatmaps[:,:,1] , linewidth=0) 
                plt.title(f'Channel 2 Target : {target_freq} Hz')
                plt.show()

                ax = sns.heatmap(target_freq_heatmaps[:,:,2] , linewidth=0) 
                plt.title(f'Channel 3 Target : {target_freq} Hz')
                plt.show()

            if heatmaps is None:
                heatmaps = target_freq_heatmaps
            else:
                heatmaps += target_freq_heatmaps

        if denoise:
            heatmaps /= noise_heatmaps
            heatmaps[heatmaps < 0] = 0  

        heatmap_chan1 = heatmaps[:,:,0]
        heatmap_chan2 = heatmaps[:,:,1]
        heatmap_chan3 = heatmaps[:,:,2]

        if display:
            ax = sns.heatmap(heatmap_chan1, linewidth=0)
            plt.title('Final Channel 1')
            plt.show()

            ax = sns.heatmap(heatmap_chan2, linewidth=0)
            plt.title('Final Channel 2')
            plt.show()

            ax = sns.heatmap(heatmap_chan3, linewidth=0) 
            plt.title('Final Channel 3')
            plt.show()

        png_heatmap_chan1 = minmax_scale(heatmap_chan1.flatten()).reshape(heatmap_chan1.shape) * 255
        png_heatmap_chan1 = png_heatmap_chan1.astype("uint8")
        png_heatmap_chan2 = minmax_scale(heatmap_chan2.flatten()).reshape(heatmap_chan2.shape) * 255
        png_heatmap_chan2 = png_heatmap_chan2.astype("uint8")
        png_heatmap_chan3 = minmax_scale(heatmap_chan3.flatten()).reshape(heatmap_chan3.shape) * 255
        png_heatmap_chan3 = png_heatmap_chan3.astype("uint8")
        
        return png_heatmap_chan1, png_heatmap_chan2, png_heatmap_chan3


def create_localization_heatmap(video_path, 
            frame_range = None, denoise = True, 
            display = False, output_folder = None, output_name = None):
    

    target_freqs = [config.localization_frequency] # currently just consider the localization freeuency itself. keep in list form in case want to consider harmonics or additional frequencies in future.
    
    if not os.path.exists(video_path):
        print(f"Video path {video_path} does not exist.")
        return None, None, None
    print(f"Creating heatmap from video: {video_path}")
    cap_temp = cv2.VideoCapture(video_path) 

    # calculate duration of the video (sanity check)
    frames = cap_temp.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = cap_temp.get(cv2.CAP_PROP_FPS) 
    seconds = round(frames / fps) 
    print(f"Video duration: {seconds} seconds") 

    W, H = cap_temp.get(3), cap_temp.get(4)
    print("Original video resolution (W, H): {}, {}".format(W, H))
    if W*H > config.max_heatmap_pixels: 
        #determine number of pyramidal downsamplessneed to get resolution to acceptable number
        downsamples = 1
        new_W = W / 2
        new_H = H / 2
        while (new_W / 2) * (new_H / 2) >  config.max_heatmap_pixels:
            downsamples += 1
            new_W /= 2
            new_H /= 2
    else:
        downsamples = None
    print(f"Downsampling video by {downsamples} to get number of pixels to analyze under {config.max_heatmap_pixels}.")
    

    images, fps =  loadVideo(video_path, colorspace = config.colorspace, downsamples = downsamples)
    print("Video FPS: ", fps)

    # use only a portion of the video if frame_range is specified
    if frame_range is not None:
        images = images[frame_range[0]:frame_range[1]]

    heatmap_chan1, heatmap_chan2, heatmap_chan3 = generate_heatmap(images, fps, 
                                                                    target_freqs, config.lower_range_start, config.lower_range_end, 
                                                                    config.upper_range_start, config.upper_range_end,
                                                                    config.target_lower_epsilon, config.target_upper_epsilon,
                                                                    config.target_channel, denoise, display
                                                    )
    # upsample heatmaps back to original resolution. Without this step, we are unable to apply
    # the homography established during calibration to og resolution core unite camera frames
    print(f"Upsampling heatmaps by {downsamples} to get back to original resolution")
    for d in range(downsamples):
        heatmap_chan1 = cv2.pyrUp(heatmap_chan1)
        heatmap_chan2 = cv2.pyrUp(heatmap_chan2)
        heatmap_chan3 = cv2.pyrUp(heatmap_chan3)
    
    if output_folder and output_name:
        if config.target_channel == "sum":
            cv2.imwrite(f'{output_folder}/heatmap_summed_{output_name}_{config.colorspace}.png', heatmap_chan1)
        else:
            cv2.imwrite(f'{output_folder}/heatmap_chan1_{output_name}_{config.colorspace}.png', heatmap_chan1)
            cv2.imwrite(f'{output_folder}/heatmap_chan2_{output_name}_{config.colorspace}.png', heatmap_chan2)
            cv2.imwrite(f'{output_folder}/heatmap_chan3_{output_name}_{config.colorspace}.png', heatmap_chan3)
    
    return heatmap_chan1, heatmap_chan2, heatmap_chan3



def create_calibration_heatmap(img_dir, start_image_i, end_image_i, fps, 
                               calibration_frequency, calibration_lower_range_start, calibration_lower_range_end,
                               calibration_upper_range_start, calibration_upper_range_end, 
                               calibration_target_lower_epsilon, calibration_target_upper_epsilon, 
                               calibration_target_channel, colorspace, 
                               denoise = True, display = False,
                               output_folder = None, output_name = None):


    target_freqs = [calibration_frequency] # currently just consider the calibration frequency itself. keep in list form in case want to consider harmonics or additional frequencies in future.

    dummy = np.load("{}/{}.npy".format(img_dir, start_image_i))
    H, W, _ = dummy.shape
    print("Input dimensions (H, W): ", H, W)
    if W * H > config.max_heatmap_pixels: 
        downsamples = 1
        new_W = W / 2
        new_H = H / 2
        while (new_W / 2) * (new_H / 2) >  config.max_heatmap_pixels:
            downsamples += 1
            new_W /= 2
            new_H /= 2
    images = []
    for i in range(start_image_i, end_image_i + 1):
        frame = np.load("{}/{}.npy".format(img_dir, i))
        if colorspace == "ycrcb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        for d in range(downsamples):
            frame = cv2.pyrDown(frame)
        images.append(frame)
    images = np.asarray(images)
   

    heatmap_chan1, heatmap_chan2, heatmap_chan3 = generate_heatmap(images, fps, 
                        target_freqs, calibration_lower_range_start, calibration_lower_range_end, 
                        calibration_upper_range_start, calibration_upper_range_end, 
                        calibration_target_lower_epsilon, calibration_target_upper_epsilon, 
                        calibration_target_channel, denoise, display
                     )

    # upsample heatmaps back to original resolution. Without this step, we are unable to apply
    # the homography established during calibration to og resolution core unite camera frames
    print(f"Upsampling heatmaps by {downsamples} to get back to original resolution")
    for d in range(downsamples):
        heatmap_chan1 = cv2.pyrUp(heatmap_chan1)
        heatmap_chan2 = cv2.pyrUp(heatmap_chan2)
        heatmap_chan3 = cv2.pyrUp(heatmap_chan3)

    if output_folder and output_name:
        if calibration_target_channel == "sum":
            cv2.imwrite(f'{output_folder}/heatmap_summed_{output_name}_{colorspace}.png', heatmap_chan1)
        else:
            cv2.imwrite(f'{output_folder}/heatmap_chan1_{output_name}_{colorspace}.png', heatmap_chan1)
            cv2.imwrite(f'{output_folder}/heatmap_chan2_{output_name}_{colorspace}.png', heatmap_chan2)
            cv2.imwrite(f'{output_folder}/heatmap_chan3_{output_name}_{colorspace}.png', heatmap_chan3)
        
    return heatmap_chan1, heatmap_chan2, heatmap_chan3

