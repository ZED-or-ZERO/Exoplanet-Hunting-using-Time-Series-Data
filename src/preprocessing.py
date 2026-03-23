import numpy as np
from scipy.signal import savgol_filter

def detrend_light_curve(flux_array, window_length=101, polyorder=3):
    """Removes the low-frequency trend from the light curve."""
    trend = savgol_filter(flux_array, window_length=window_length, polyorder=polyorder)
    detrended_flux = (flux_array - trend) + 1.0
    return detrended_flux