"""
Utility functions for processing MediaPipe signals
"""
import numpy as np
import config
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

def single_feature_signal_processing(signal, resample_signal = True):
    try:
        proc_signal = interp_fill_nans(np.array(signal))     
    except Exception as e:
        return [0 for i in range(config.single_dynamic_signal_len)]
    scaler = StandardScaler()
    proc_signal = scaler.fit_transform(proc_signal.reshape(-1, 1)).reshape(-1) # important so that concat signal isn't just dominated by scale differences that inflate pearson score
    proc_signal = rolling_average(proc_signal, n = 2)
    if resample_signal:
        proc_signal = ResampleLinear1D(proc_signal, config.single_dynamic_signal_len) # downsample  
    else:
        proc_signal = proc_signal.tolist()  
    return proc_signal

def rolling_average(a, n=3):
    """
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    # BELOW LINES CHANGED AS OF DEC 2024
    ret[n - 2] = ret[n - 1]
    return ret[n - 2:] / n
    # return ret[n - 1:] / n OG 

def ResampleLinear1D(x, targetLen):
    """
    https://stackoverflow.com/questions/29085268/resample-a-numpy-array
    """
    factor = len(x) / targetLen
    n = np.ceil(x.size / factor).astype(int)
    f = interp1d(np.linspace(0, 1, x.size), x, 'linear')
    return f(np.linspace(0, 1, n)).tolist()


def interp_fill_nans(signal):
    """
    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """
    # if np.count_nonzero(np.isnan(signal)) > 0:
        # print("Interping to fill NaNs...")
    ok = ~np.isnan(signal)
    xp = ok.ravel().nonzero()[0]
    fp = signal[~np.isnan(signal)]
    x  = np.isnan(signal).ravel().nonzero()[0]
    signal[np.isnan(signal)] = np.interp(x, xp, fp)
    return signal
