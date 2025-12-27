import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np #type: ignore
from utils.bci_pipe import extract_features

def classify_mood(features):
    """
    The 'Decision Maker'.
    Uses the spatial features to guess the mood.
    """
    # Extract powers (averaging front vs back channels)
    # Front: 0, 1 | Back: 2, 3
    front_beta = np.mean(features['beta'][:2])
    back_alpha = np.mean(features['alpha'][2:])
    front_noise = np.mean(features['beta'][:2]) # High beta often correlates with noise/tension
    
    # Calculate Ratios
    # Focus Ratio: Beta / Theta
    focus_score = front_beta / (np.mean(features['theta'][:2]) + 1e-6)
    
    # Relaxation Ratio: Alpha / Beta
    relax_score = back_alpha / (np.mean(features['beta'][2:]) + 1e-6)
    
    print(f"DEBUG: Focus Score: {focus_score:.2f} | Relax Score: {relax_score:.2f}")

    # --- CLASSIFICATION LOGIC ---
    
    # 1. Check for Anger (High Frontal Beta/Noise + Low Alpha)
    if focus_score > 1.5 and relax_score < 1.0:
        return "anger" # Or high stress/focus (hard to distinguish without more sensors)
        
    # 2. Check for Sadness (Dominant Back Alpha + Low Beta)
    elif relax_score > 2.0:
        return "sad"
        
    # 3. Check for Happiness (High Alpha AND High Beta - "Active Calm")
    elif relax_score > 1.0 and focus_score > 0.8:
        return "happy"
        
    else:
        return "neutral"

# --- TEST THE PIPELINE ---
if __name__ == "__main__":
    from data.brain import get_multichannel_eeg
    # Generate fresh "Sad" data
    print("--- Simulating SAD Brain ---")
    raw_eeg, fs = get_multichannel_eeg(mood="sad")
    feats = extract_features(raw_eeg, fs)
    detected = classify_mood(feats)
    print(f"Pipeline Result: {detected.upper()}")
    
    print("\n--- Simulating ANGER Brain ---")
    raw_eeg, fs = get_multichannel_eeg(mood="anger")
    feats = extract_features(raw_eeg, fs)
    detected = classify_mood(feats)
    print(f"Pipeline Result: {detected.upper()}")