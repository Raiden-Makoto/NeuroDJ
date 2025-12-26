import numpy as np #type: ignore
from scipy.signal import butter, filtfilt, welch #type: ignore

def bandpass_filter(data, fs, lowcut=1.0, highcut=50.0):
    """
    The 'Brillo Pad'.
    Removes signals below 1Hz (slow drift) and above 50Hz (electrical hum/muscle noise).
    """
    nyquist = 0.5 * fs # nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    
    # filtfilt applies the filter forward and backward to avoid phase shift
    return filtfilt(b, a, data, axis=-1)

def extract_features(eeg_matrix, fs):
    """
    The 'Translator'.
    Converts raw voltage -> Band Power (Alpha, Beta, Theta).
    Returns a dictionary of powers per channel.
    """
    # 1. Apply Filter first!
    clean_data = bandpass_filter(eeg_matrix, fs)
    
    # 2. Get Power Spectral Density (PSD) using Welch's method
    # This turns Time Domain -> Frequency Domain
    freqs, psd = welch(clean_data, fs, nperseg=fs*2, axis=-1)
    
    # 3. Average the power in specific bands
    # Define bands: Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz)
    bands = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    features = {}
    
    for band_name, (f_min, f_max) in bands.items():
        # Find indices of frequencies in this band
        idx = np.where((freqs >= f_min) & (freqs <= f_max))[0]
        
        # Mean power in this band for ALL channels
        # Shape of psd is (4, frequencies)
        band_power = np.mean(psd[:, idx], axis=1)
        features[band_name] = band_power 

    return features