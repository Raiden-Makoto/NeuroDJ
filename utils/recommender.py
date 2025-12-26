import numpy as np #type: ignore
from data.fake_playlist import DUMMY_PLAYLIST, USER_TASTE_PROFILE

def find_closest_song(suggested_v, suggested_e):
    """
    Finds the 'Nearest Neighbor' in the dummy database.
    This is exactly how Spotify 'Radio' works.
    """
    best_song = None
    min_dist = float('inf')
    
    for song in DUMMY_PLAYLIST:
        # Euclidean distance
        dist = np.sqrt((song['valence'] - suggested_v)**2 + 
                       (song['energy'] - suggested_e)**2)
        if dist < min_dist:
            min_dist = dist
            best_song = song
            
    return best_song

def get_user_reaction(song, current_mood):
    """
    Does the user like the song picked?
    """
    target = USER_TASTE_PROFILE[current_mood]
    
    # Distance between the SONG and the USER'S DESIRE
    dist = np.sqrt((song['valence'] - target['target_v'])**2 + 
                   (song['energy'] - target['target_e'])**2)
    
    # If distance is small (< 0.2), they probably like it
    # We add randomness to make it realistic
    prob_like = np.exp(-5 * dist) 
    return 1 if np.random.random() < prob_like else 0