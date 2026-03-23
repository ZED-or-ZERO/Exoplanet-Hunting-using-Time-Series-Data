import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from .preprocessing import detrend_light_curve 

def compute_fft(flux_array, time_interval):
    """Calculates the FFT and returns the power spectrum."""
    N = len(flux_array)
    fft_values = fft(flux_array)
    frequencies = fftfreq(N, d=time_interval)
    
    positive_freqs = frequencies[1:N//2]
    power_spectrum = np.abs(fft_values[1:N//2])
    
    return positive_freqs, power_spectrum


def generate_fft_dataset(flux_df, time_df, n_features=500):
    """
    Iterates through all the stars, applies detrending and FFT.
    Limit the number of features (n_features) so as not to overload the memory.
    """
    print("Starting Feature Extraction (FFT)...")
    fft_features = []
    
    # Iterating along the lines of the dataframe
    for idx in range(len(flux_df)):
        raw_flux = flux_df.drop(columns=['id', 'class']).iloc[idx].values
        time_vals = time_df.drop(columns=['id', 'class']).iloc[idx].values
        
        # 1. Detrending
        detrended_flux = detrend_light_curve(raw_flux)
        
        # 2. Calculating the time step (dt)
        dt = time_vals[1] - time_vals[0]
        
        # 3. FFT
        freqs, power = compute_fft(detrended_flux, dt)
        
        # Take only the first n_features of frequencies (interested in low frequencies for planets)
        # Save the power spectrum as a feature vector
        fft_features.append(power[:n_features])
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(flux_df)} stars...")

    # Creating a new DataFrame from the received lists
    feature_cols = [f'freq_{i+1}' for i in range(n_features)]
    fft_df = pd.DataFrame(fft_features, columns=feature_cols)
    
    # Putting the id and class back in place
    fft_df.insert(0, 'id', flux_df['id'])
    fft_df.insert(1, 'class', flux_df['class'])
    
    print("Feature Extraction Complete. New shape:", fft_df.shape)
    return fft_df