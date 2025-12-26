import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
from scipy.signal import butter, filtfilt #type: ignore

def generate_pink_noise(num_points):
    """
    Generates 1/f noise (Pink Noise).
    Real brain background activity follows a 1/f power law, 
    meaning low frequencies have much higher amplitude than high ones.
    """
    # 1. Generate white noise
    white = np.random.normal(0, 1, num_points)
    
    # 2. FFT to frequency domain
    X_white = np.fft.rfft(white)
    
    # 3. Scale amplitudes by 1/f
    frequencies = np.fft.rfftfreq(num_points)
    # Avoid division by zero at f=0
    scale = 1 / (frequencies + 1e-6)
    # Apply 1/sqrt(f) to magnitude to get 1/f power spectrum
    X_pink = X_white * np.sqrt(scale)
    
    # 4. Inverse FFT back to time domain
    pink_noise = np.fft.irfft(X_pink)
    
    # Normalize to standard EEG voltage range (microvolts)
    # Real EEG is usually +/- 50 to 100 uV
    pink_noise = pink_noise / np.std(pink_noise) * 10
    
    # Match length (irfft can be off by 1 point)
    return pink_noise[:num_points]

def add_wave(signal, fs, freq, amp):
    # Simplified wave adder for multiple channels
    t = np.arange(len(signal)) / fs
    wave = amp * np.sin(2 * np.pi * freq * t)
    # Add random phase shift so channels aren't perfectly identical (unrealistic)
    phase = np.random.uniform(0, 2*np.pi)
    wave = amp * np.sin(2 * np.pi * freq * t + phase)
    return signal + wave