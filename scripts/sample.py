import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.recommender import find_closest_song
from data.brain import get_multichannel_eeg
from data.fake_playlist import DUMMY_PLAYLIST
import numpy as np #type: ignore
from bayes_opt import BayesianOptimization #type: ignore
from bayes_opt.acquisition import UpperConfidenceBound #type: ignore

def run_neuro_dj_session():
    print("🎧 STARTING NEURO-DJ WITH DUMMY DATABASE...")
    
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
    print(f"✅ Brain wave simulation complete ({raw_eeg.shape[1] / fs:.1f} seconds of data)")
    
    # B. OPTIMIZATION LOOP
    print(f"\n--- 3. DJ SPINNING TRACKS FOR: {detected_mood.upper()} ---")
    
    acquisition = UpperConfidenceBound(kappa=2.5)
    
    optimizer = BayesianOptimization(
        f=None, # We manually probe, so f is None
        pbounds={'valence': (0, 1), 'energy': (0, 1)},
        acquisition_function=acquisition,
        verbose=0,
        random_state=42,
        allow_duplicate_points=True
    )
    
    print(f"\n{'SONG TITLE':<25} | {'ARTIST':<15} | {'V / E':<10}")
    print("-" * 55)
    
    # Taboo list: tracks songs we've already played to prevent infinite loops
    taboo_list = set()  # Store (title, artist) tuples
    
    def find_closest_song_excluding_taboo(suggested_v, suggested_e, taboo):
        """Find closest song that's not in the taboo list"""
        best_song = None
        min_dist = float('inf')
        
        for song in DUMMY_PLAYLIST:
            song_key = (song['title'], song['artist'])
            if song_key in taboo:
                continue  # Skip taboo songs
            
            # Euclidean distance
            dist = np.sqrt((song['valence'] - suggested_v)**2 + 
                          (song['energy'] - suggested_e)**2)
            if dist < min_dist:
                min_dist = dist
                best_song = song
                
        return best_song
    
    i = 0
    while i < 10:
        # 1. AI Suggests ideal coordinates
        next_point = optimizer.suggest()
        
        # 2. Database Retrieval (Find closest song excluding taboo list)
        song = find_closest_song_excluding_taboo(
            next_point['valence'], 
            next_point['energy'],
            taboo_list
        )
        
        # 3. Check if we've run out of songs
        if song is None:
            print("\nLooping the playlist")
            break
        
        # 4. Display song and get user reaction
        ve_str = f"{song['valence']:.2f}/{song['energy']:.2f}"
        print(f"\n[{i+1}/10] Now playing: {song['title']} by {song['artist']} (V/E: {ve_str})")
        
        # 5. Get interactive user feedback
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
        
        # 6. Add song to taboo list (prevent repeats)
        song_key = (song['title'], song['artist'])
        taboo_list.add(song_key)
        
        # 7. TEACH THE AI
        # Critical: We teach it the parameters of the SONG we played, 
        # not the parameters it suggested. This reduces noise.
        optimizer.register(
            params={'valence': song['valence'], 'energy': song['energy']},
            target=feedback
        )
        
        i += 1

    # C. SUMMARY
    print("-" * 65)
    print("🎉 SESSION COMPLETE")
    # Find which song is closest to the AI's final "Best" belief
    best_coords = optimizer.max['params']
    final_pick = find_closest_song(best_coords['valence'], best_coords['energy'])
    print(f"The AI concluded the perfect song for you is: {final_pick['title']} by {final_pick['artist']}")

if __name__ == "__main__":
    run_neuro_dj_session()