import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import extract_features
from data.brain import get_multichannel_eeg
from scripts.classifier import classify_mood
from utils.recommender import find_closest_song, get_user_reaction
from bayes_opt import BayesianOptimization, UtilityFunction #type: ignore

def run_neuro_dj_session():
    print("🎧 STARTING NEURO-DJ WITH DUMMY DATABASE...")
    
    # A. DETECT MOOD (Simulated)
    # Let's force "ANGER" to see if it finds Metallica
    true_mood = "anger"
    print(f"\n--- 1. READING BRAINWAVES (True State: {true_mood.upper()}) ---")
    
    raw_eeg, fs = get_multichannel_eeg(mood=true_mood)
    features = extract_features(raw_eeg, fs)
    detected_mood = classify_mood(features)
    
    print(f"✅ DETECTED: {detected_mood.upper()}")
    
    # B. OPTIMIZATION LOOP
    print(f"\n--- 2. DJ SPINNING TRACKS FOR: {detected_mood.upper()} ---")
    
    optimizer = BayesianOptimization(
        f=None, # We manually probe, so f is None
        pbounds={'valence': (0, 1), 'energy': (0, 1)},
        verbose=0,
        random_state=42
    )
    
    utility = UtilityFunction(kind="ucb", kappa=2.5)
    
    print(f"{'SONG TITLE':<20} | {'ARTIST':<12} | {'V / E':<10} | {'REACTION'}")
    print("-" * 65)
    
    for i in range(10):
        # 1. AI Suggests ideal coordinates
        next_point = optimizer.suggest(utility)
        
        # 2. Database Retrieval (Find closest song)
        song = find_closest_song(next_point['valence'], next_point['energy'])
        
        # 3. User Reaction (To the actual song)
        feedback = get_user_reaction(song, detected_mood)
        
        # 4. Display
        reaction_emoji = "❤️  LIKED" if feedback == 1 else "❌  SKIPPED"
        ve_str = f"{song['valence']:.2f}/{song['energy']:.2f}"
        print(f"{song['title']:<20} | {song['artist']:<12} | {ve_str:<10} | {reaction_emoji}")
        
        # 5. TEACH THE AI
        # Critical: We teach it the parameters of the SONG we played, 
        # not the parameters it suggested. This reduces noise.
        try:
            optimizer.register(
                params={'valence': song['valence'], 'energy': song['energy']},
                target=feedback
            )
        except KeyError:
            pass # Creating duplicate points happens in small datasets

    # C. SUMMARY
    print("-" * 65)
    print("🎉 SESSION COMPLETE")
    # Find which song is closest to the AI's final "Best" belief
    best_coords = optimizer.max['params']
    final_pick = find_closest_song(best_coords['valence'], best_coords['energy'])
    print(f"The AI concluded the perfect song for you is: {final_pick['title']} by {final_pick['artist']}")

if __name__ == "__main__":
    run_neuro_dj_session()