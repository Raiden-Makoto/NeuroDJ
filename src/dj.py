import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np #type: ignore
from bayes_opt import BayesianOptimization #type: ignore
from bayes_opt.acquisition import UpperConfidenceBound #type: ignore

from data.brain import get_multichannel_eeg
from utils import extract_features
from scripts.classifier import classify_mood
from scripts.spotify import OfflineSpotifyBrain

# --- 1. DEFINE THE 4D TASTE PROFILE ---
# This acts as the "Ground Truth" describing what the user wants in each state.
USER_TASTE_PROFILE = {
    "sad": {
        "valence": 0.2, 
        "energy": 0.2, 
        "acousticness": 0.9, # High acoustic = Folk/Piano
        "liveness": 0.1 
    },
    "happy": {
        "valence": 0.9, 
        "energy": 0.8, 
        "acousticness": 0.05, # Low acoustic = Pop production
        "liveness": 0.1 
    },
    "anger": {
        "valence": 0.1, 
        "energy": 0.9, 
        "acousticness": 0.01, # Electronic/Distorted
        "liveness": 0.3       # Slightly rawer sound
    },
    "neutral": {
        "valence": 0.5, "energy": 0.5, "acousticness": 0.5, "liveness": 0.1
    }
}

def get_user_feedback(song_features, current_mood):
    """
    Calculates distance between the SONG'S features and the USER'S desire.
    Now operates in 4 Dimensions.
    """
    target = USER_TASTE_PROFILE[current_mood]
    
    # 4D Euclidean Distance
    dist = np.sqrt(
        (song_features['valence'] - target['valence'])**2 + 
        (song_features['energy'] - target['energy'])**2 +
        (song_features['acousticness'] - target['acousticness'])**2 +
        (song_features['liveness'] - target['liveness'])**2
    )
    
    # Sigmoid function to determine probability of a "Like"
    # If distance is 0 (perfect match), prob is 1.0
    # If distance is > 0.5, prob drops rapidly
    prob_like = np.exp(-4 * dist) 
    
    return 1 if np.random.random() < prob_like else 0

# --- 2. THE MAIN LOOP ---
def run_neuro_dj_session():
    print("🧠 BOOTING NEURO-DJ (4-DIMENSIONAL MODE)...")
    
    # A. GET USER MOOD AND GENERATE BRAIN WAVES
    valid_moods = ["sad", "happy", "anger", "neutral"]
    print(f"\n--- 1. WHAT'S YOUR CURRENT MOOD? ---")
    print("Available moods: sad, happy, anger, neutral")
    
    while True:
        user_mood = input("Enter your mood: ").strip().lower()
        if user_mood in valid_moods:
            detected_mood = user_mood
            break
        else:
            print(f"Invalid mood. Please choose from: {', '.join(valid_moods)}")
    
    print(f"✅ SELECTED MOOD: {detected_mood.upper()}")
    
    # Generate brain waves based on the user's mood
    print(f"\n--- 2. GENERATING BRAIN WAVES FOR {detected_mood.upper()} MOOD ---")
    raw_eeg, fs = get_multichannel_eeg(mood=detected_mood)
    features = extract_features(raw_eeg, fs)
    detected_mood_from_eeg = classify_mood(features)
    
    print(f"✅ Brain wave simulation complete ({raw_eeg.shape[1] / fs:.1f} seconds of data)")
    print(f"✅ DETECTED MOOD FROM EEG: {detected_mood_from_eeg.upper()}")
    
    # Use the detected mood for the session
    detected_mood = detected_mood_from_eeg
    
    # B. INITIALIZE DATABASE
    print(f"\n--- 3. LOADING MUSIC DATABASE ---")
    dj_brain = OfflineSpotifyBrain("data/taylor_swift.csv")
    
    # C. OPTIMIZATION LOOP
    print(f"\n--- 4. DJ SPINNING TRACKS FOR: {detected_mood.upper()} ---")
    
    acquisition = UpperConfidenceBound(kappa=2.5)
    
    optimizer = BayesianOptimization(
        f=None, # We manually probe, so f is None
        pbounds={
            'valence': (0, 1), 
            'energy': (0, 1),
            'acousticness': (0, 1),
            'liveness': (0, 1)
        },
        acquisition_function=acquisition,
        verbose=0,
        random_state=42,
        allow_duplicate_points=True
    )
    
    print(f"\n{'SONG TITLE':<25} | {'ALBUM':<15} | {'V / E / A':<15} | {'REACTION'}")
    print("-" * 80)
    
    for i in range(10): # Run 10 iterations
        # 1. AI Suggests 4 coordinates
        next_point = optimizer.suggest()
        
        # 2. Database Retrieval (Find closest song in CSV)
        song = dj_brain.get_track_from_features(next_point)
        
        if not song:
            print("❌ No song found (Library exhausted).")
            break
            
        # 3. Display song and get user reaction
        # Extract features from the song's features dict
        # The features dict contains the actual values from the CSV
        song_features = song.get('features', {})
        # Get features with fallbacks (should all be present since we search for them)
        valence = song_features.get('valence', next_point.get('valence', 0.5))
        energy = song_features.get('energy', next_point.get('energy', 0.5))
        acousticness = song_features.get('acousticness', next_point.get('acousticness', 0.5))
        liveness = song_features.get('liveness', next_point.get('liveness', 0.1))
        
        stats_str = f"{valence:.2f}/{energy:.2f}/{acousticness:.2f}"
        
        # Truncate title for clean display
        title_display = (song['title'][:23] + '..') if len(song['title']) > 23 else song['title']
        album_display = (song['album'][:13] + '..') if len(song['album']) > 13 else song['album']
        
        print(f"\n[{i+1}/10] Now playing: {song['title']} (Album: {song['album']})")
        print(f"  Features: V={valence:.2f}, E={energy:.2f}, A={acousticness:.2f}, L={liveness:.2f}")
        
        # 4. Get interactive user feedback
        while True:
            user_input = input("Do you like this song? (y/n): ").strip().lower()
            if user_input == 'y':
                feedback = 1
                reaction_emoji = "❤️  LIKED"
                break
            elif user_input == 'n':
                feedback = 0
                reaction_emoji = "❌  SKIPPED"
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        
        print(f"  → {reaction_emoji}")
        
        # 5. TEACH THE AI
        # Critical: We teach it the parameters of the SONG we played, 
        # not the parameters it suggested. This reduces noise.
        real_song_features = {
            'valence': valence,
            'energy': energy,
            'acousticness': acousticness,
            'liveness': liveness
        }
        
        optimizer.register(
            params=real_song_features,
            target=feedback
        )
        
        # Display in table format
        print(f"{title_display:<25} | {album_display:<15} | {stats_str:<15} | {reaction_emoji}")

    # D. SUMMARY
    print("-" * 80)
    print("🎉 SESSION COMPLETE")
    best_params = optimizer.max['params']
    print(f"\nThe AI learned your preference for {detected_mood.upper()}:")
    print(f"   -> Valence:      {best_params['valence']:.2f}")
    print(f"   -> Energy:       {best_params['energy']:.2f}")
    print(f"   -> Acousticness: {best_params['acousticness']:.2f}")
    print(f"   -> Liveness:     {best_params['liveness']:.2f}")
    
    # Find the best matching song based on learned preferences
    final_song = dj_brain.get_track_from_features(best_params)
    if final_song:
        print(f"\n🎵 Best match based on learned preferences:")
        print(f"   -> {final_song['title']} (Album: {final_song['album']})")
        print(f"   -> Link: {final_song['url']}")

if __name__ == "__main__":
    run_neuro_dj_session()
