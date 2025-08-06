"""
Utility functions for computing the perceptibility of projected cells/optical modulations
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import skimage
from scipy.optimize import minimize
import sys
sys.path.append('../common/')
import config

def get_on_off_cell_val_from_cell_signal(cell_signal, image_sequence, cell_boundaries, output_name = None):
    """
    the off and on cell valuse should just be the RGB value at the frame in which the cell's R + G + B is 
    minimized or maximized across the whole signal. This follows from the fact that the SLM always adds light, increasing
    the intensity in some channel(s). 
    """ 
    # to save some time, use existing cell signal to determine frame where min intensity occurs
    min_intensity = float('inf')
    min_intensity_i = 0
    max_intensity = float('-inf')
    max_intensity_i = 0
    for i, intensity in enumerate(cell_signal):
        if intensity < min_intensity:
            min_intensity = intensity
            min_intensity_i = i
        if intensity > max_intensity:
            max_intensity = intensity
            max_intensity_i = i
    
    cell_top, cell_left, cell_bottom, cell_right = cell_boundaries
    mask = np.zeros((config.slm_H, config.slm_W), np.uint8)
    cv2.rectangle(mask, (cell_left, cell_top), (cell_right, cell_bottom), 255, -1)   

    # for both min and max, apply mask and get bgr from those frames
    target_min_img = image_sequence[min_intensity_i, :,:, :]
    min_intensity_bgr = cv2.mean(target_min_img, mask=mask)[:3]

    target_max_img = image_sequence[max_intensity_i, :,:, :]
    max_intensity_bgr = cv2.mean(target_max_img, mask=mask)[:3]

    return min_intensity_bgr, max_intensity_bgr


def update_slm_bgr(curr_cell_bgr, curr_delta_min, cell_color_model, logging_cell_name = None, output_name = None):
    """
    for old approach using regression model to predict optimal SLM values from deltas
    input: current cell r, g, b from core unit camera view
    output: recommended slm values
    """
    # solve minimization problem using curr_min_delta and cell appearance
    curr_cell_b, curr_cell_g, curr_cell_r = curr_cell_bgr
    f = lambda x : CIEDE2000([curr_cell_r, curr_cell_g, curr_cell_b], [curr_cell_r + x[0], curr_cell_g + x[1], curr_cell_b + x[2]])
    cons = ({'type': 'eq', 'fun': lambda x : x[0] + x[1] + x[2] - curr_delta_min})
    res = minimize(f, (0,0,0), constraints=cons)
    d_r, d_g, d_b = res.x
    r_p, g_p, b_p = curr_cell_r + d_r, curr_cell_g + d_g, curr_cell_b + d_b

    curr_cell_b_lin, curr_cell_g_lin, curr_cell_r_lin = srgb2lin([curr_cell_b, curr_cell_g, curr_cell_r]) #srgb2lin preserves order of chanels as they are passed in
    b_p_lin, g_p_lin, r_p_lin = srgb2lin([b_p, g_p, r_p])
    d_b_lin = b_p_lin - curr_cell_b_lin
    d_g_lin = g_p_lin - curr_cell_g_lin
    d_r_lin = r_p_lin - curr_cell_r_lin

    # query multioutput regression model for this cell to get SLM rgb
    rec_slm_val = cell_color_model.predict(np.array([d_r_lin, d_g_lin, d_b_lin]).reshape(1, -1)) #predict slm value (rgb) from inputted delta
    rec_slm_val = rec_slm_val[0].astype(int)
    rec_slm_val[rec_slm_val < 0] = 0
    rec_slm_val = rec_slm_val.tolist()[::-1] #has to be int to create BMP for SLM, and also should be BGR order to be consistent with encoding code


    f = open(f"{config.trial_materials}/regression_logging.txt", "a+")
    f.write(f"{logging_cell_name} Curr delta_min: {curr_delta_min}. Curr cell bgr (lin): {curr_cell_b_lin:.2f} {curr_cell_g_lin:.2f} {curr_cell_r_lin:.2f}. Recommended bgr (lin): {b_p_lin:.2f} {g_p_lin:.2f} {r_p_lin:.2f}, w/ deltas {d_b_lin:.2f} {d_g_lin:.2f} {d_r_lin:.2f}. Rec SLM val: {rec_slm_val}\n")
    f.close()

    if output_name is not None:
        fig, axes = plt.subplots(2)
        dummy = np.zeros((config.N, config.N, 3)).astype(np.uint8)
        dummy[:,:,2] = int(curr_cell_b)
        dummy[:,:, 1] = int(curr_cell_g)
        dummy[:,:, 0] = int(curr_cell_r)
        axes[0].set_title(f"Curr nonlinear BGR: {curr_cell_b:.2f} {curr_cell_g:.2f} {curr_cell_r:.2f}\nCurr linear BGR: {curr_cell_b_lin:.2f} {curr_cell_g_lin:.2f} {curr_cell_r_lin:.2f}")
        axes[0].imshow(dummy)

        dummy[:,:,2] = int(b_p)
        dummy[:,:, 1] = int(g_p)
        dummy[:,:, 0] = int(r_p)
        axes[1].set_title(f"Recommended nonlinear BGR: {b_p:.2f} {g_p:.2f} {r_p:.2f}\nDeltas of {d_b:.2f} {d_g:.2f} {d_r:.2f}\nRecommended linear BGR: {b_p_lin:.2f} {g_p_lin:.2f} {r_p_lin:.2f}\nDeltas of {d_b_lin:.2f} {d_g_lin:.2f} {d_r_lin:.2f}\nRec SLM: {rec_slm_val}")
        axes[1].imshow(dummy)

        plt.savefig(f"{config.trial_materials}/perceptibility/{output_name}.png")
        plt.clf()

    return rec_slm_val

def srgb2lin(rgb):
    """
    https://www.cyril-richon.com/blog/2019/1/23/python-srgb-to-linear-linear-to-srgb
    """
    lin = []
    for c in rgb:
        c = float(c)
        c /= 255
        if c <= 0.0404482362771082:
            c = c / 12.92
        else:
            c = pow(((c + 0.055) / 1.055), 2.4)
        c *= 255
        lin.append(c)
    return lin


def CIEDE2000(sRGB1, sRGB2):
    """
    Calculates CIEDE2000 color distance between two sRGB colors
    From https://github.com/lovro-i/CIEDE2000/blob/master/ciede2000.py, with addition of initial conversion 
    from srgb to lab
    """
    Lab_1 = list(skimage.color.rgb2lab([[[sRGB1[0] / 255, sRGB1[1] / 255, sRGB1[2] / 255]]])[0][0])
    Lab_2 = list(skimage.color.rgb2lab([[[sRGB2[0] / 255, sRGB2[1] / 255, sRGB2[2] / 255]]])[0][0])

    C_25_7 = 6103515625 # 25**7
    
    L1, a1, b1 = Lab_1[0], Lab_1[1], Lab_1[2]
    L2, a2, b2 = Lab_2[0], Lab_2[1], Lab_2[2]
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_ave**7 / (C_ave**7 + C_25_7)))
    
    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2
    
    C1_ = math.sqrt(a1_**2 + b1_**2)
    C2_ = math.sqrt(a2_**2 + b2_**2)
    
    if b1_ == 0 and a1_ == 0: h1_ = 0
    elif a1_ >= 0: h1_ = math.atan2(b1_, a1_)
    else: h1_ = math.atan2(b1_, a1_) + 2 * math.pi
    
    if b2_ == 0 and a2_ == 0: h2_ = 0
    elif a2_ >= 0: h2_ = math.atan2(b2_, a2_)
    else: h2_ = math.atan2(b2_, a2_) + 2 * math.pi

    dL_ = L2_ - L1_
    dC_ = C2_ - C1_    
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0: dh_ = 0
    elif dh_ > math.pi: dh_ -= 2 * math.pi
    elif dh_ < -math.pi: dh_ += 2 * math.pi        
    dH_ = 2 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2)
    
    L_ave = (L1_ + L2_) / 2
    C_ave = (C1_ + C2_) / 2
    
    _dh = abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_
    
    if _dh <= math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2
    elif _dh  > math.pi and _sh < 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 + math.pi
    elif _dh  > math.pi and _sh >= 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 - math.pi 
    else: h_ave = h1_ + h2_
    
    T = 1 - 0.17 * math.cos(h_ave - math.pi / 6) + 0.24 * math.cos(2 * h_ave) + 0.32 * math.cos(3 * h_ave + math.pi / 30) - 0.2 * math.cos(4 * h_ave - 63 * math.pi / 180)
    
    h_ave_deg = h_ave * 180 / math.pi
    if h_ave_deg < 0: h_ave_deg += 360
    elif h_ave_deg > 360: h_ave_deg -= 360
    dTheta = 30 * math.exp(-(((h_ave_deg - 275) / 25)**2))
    
    R_C = 2 * math.sqrt(C_ave**7 / (C_ave**7 + C_25_7))  
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T
    
    Lm50s = (L_ave - 50)**2
    S_L = 1 + 0.015 * Lm50s / math.sqrt(20 + Lm50s)
    R_T = -math.sin(dTheta * math.pi / 90) * R_C

    k_L, k_C, k_H = 1, 1, 1
    
    f_L = dL_ / k_L / S_L
    f_C = dC_ / k_C / S_C
    f_H = dH_ / k_H / S_H
    
    dE_00 = math.sqrt(f_L**2 + f_C**2 + f_H**2 + R_T * f_C * f_H)
    return dE_00