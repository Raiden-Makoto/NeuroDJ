import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import generate_pink_noise, add_wave
import numpy as np #type: ignore

def get_multichannel_eeg(mood="neutral", duration_sec=10):
    fs = 256
    n_points = duration_sec * fs
    n_channels = 4 # Emulating a Muse Headset (AF7, AF8, TP9, TP10)
    
    # Initialize empty matrix (Channels x Time)
    eeg_matrix = np.zeros((n_channels, n_points))
    
    # 1. Base Noise (Independent per channel)
    for ch in range(n_channels):
        eeg_matrix[ch] = generate_pink_noise(n_points)
        
    # 2. Add Mood Signatures (Spatially specific!)
    
    if mood == "sad":
        # Sadness = High Theta/Alpha (Stronger in BACK channels 2 & 3)
        for ch in [2, 3]: 
            eeg_matrix[ch] = add_wave(eeg_matrix[ch], fs, freq=6, amp=20) # Theta
            eeg_matrix[ch] = add_wave(eeg_matrix[ch], fs, freq=10, amp=15) # Alpha
        # Front channels get weaker signal
        for ch in [0, 1]:
            eeg_matrix[ch] = add_wave(eeg_matrix[ch], fs, freq=6, amp=5)

    elif mood == "anger":
        # Anger = Beta + Muscle Noise (Stronger in FRONT channels 0 & 1 due to jaw clench)
        for ch in [0, 1]:
            # High Beta
            eeg_matrix[ch] = add_wave(eeg_matrix[ch], fs, freq=25, amp=15)
            # Massive Muscle Noise (EMG)
            eeg_matrix[ch] += np.random.normal(0, 25, n_points) 
        
        for ch in [2, 3]:
            # Back channels hear less muscle noise
            eeg_matrix[ch] = add_wave(eeg_matrix[ch], fs, freq=25, amp=10)
            eeg_matrix[ch] += np.random.normal(0, 5, n_points)

    elif mood == "happy":
        # Happy = Balanced Alpha/Beta (Global synchrony)
        for ch in range(4):
            eeg_matrix[ch] = add_wave(eeg_matrix[ch], fs, freq=12, amp=15) # High Alpha
            eeg_matrix[ch] = add_wave(eeg_matrix[ch], fs, freq=20, amp=10) # Beta

    # 3. Add Blinks (Only affect Frontal Channels 0 & 1)
    # This is a key "realism" detail. Blinks barely show up on the back of head.
    if np.random.random() > 0.5:
        t = np.arange(n_points) / fs
        blink_time = np.random.uniform(1, 9)
        artifact = 150 * np.exp(-0.5 * ((t - blink_time) / 0.1)**2)
        
        eeg_matrix[0] += artifact # Front Left
        eeg_matrix[1] += artifact # Front Right
        eeg_matrix[2] += artifact * 0.1 # Back Left (faint echo)
        eeg_matrix[3] += artifact * 0.1 # Back Right (faint echo)

    return eeg_matrix, fs