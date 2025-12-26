import pandas as pd #type: ignore
import numpy as np #type: ignore

class OfflineSpotifyBrain:
    def __init__(self, csv_path="data/taylor_swift.csv"):
        print(f"📂 Loading Discography from {csv_path}...")
        
        try:
            self.df = pd.read_csv(csv_path)
            # Normalize column names to lowercase just in case
            self.df.columns = [c.lower() for c in self.df.columns]
            
            # Verify we have the available columns
            available_cols = ['valence', 'energy', 'acousticness', 'liveness', 'loudness']
            missing = [c for c in available_cols if c not in self.df.columns]
            
            if missing:
                print(f"⚠️ WARNING: Dataset is missing columns: {missing}")
                print("   -> We will ignore these features during search.")
            else:
                print(f"✅ Available features: {', '.join(available_cols)}")
            
            print(f"✅ Loaded {len(self.df)} tracks!")
            self.taboo_list = set()
            
        except FileNotFoundError:
            print(f"❌ ERROR: File not found: {csv_path}")
            self.df = pd.DataFrame()

    def get_track_from_features(self, target_dict):
        """
        target_dict: A dictionary of features we want to optimize for.
        e.g., {'valence': 0.2, 'energy': 0.1, 'instrumentalness': 0.9}
        """
        if self.df.empty: return None

        # 1. Filter Taboo (Don't repeat songs)
        candidates = self.df[~self.df['track_name'].isin(self.taboo_list)].copy()
        
        if candidates.empty:
            print("⚠️ Played all songs! Resetting library.")
            self.taboo_list.clear()
            candidates = self.df.copy()

        # 2. Calculate 5-Dimensional Distance
        # We start with 0 distance and add the squared error for each requested feature
        candidates['distance'] = 0.0
        
        for feature, target_value in target_dict.items():
            if feature in candidates.columns:
                # Add (Actual - Target)^2
                candidates['distance'] += (candidates[feature] - target_value) ** 2
            else:
                # If column is missing in CSV, skip it silently
                pass
        
        # 3. Final Square Root (Standard Euclidean Distance)
        candidates['distance'] = np.sqrt(candidates['distance'])
        
        # 4. Find Best Match
        best_song = candidates.loc[candidates['distance'].idxmin()]
        self.taboo_list.add(best_song['track_name'])
        
        # 5. Build Result
        # Create a search link since we don't have direct URLs
        clean_title = best_song['track_name'].replace(" ", "%20")
        search_url = f"https://open.spotify.com/search/{clean_title}%20Taylor%20Swift"

        # Extract feature values that were used in the search
        features_found = {}
        for feature in target_dict.keys():
            if feature in best_song.index:
                features_found[feature] = best_song[feature]

        return {
            'title': best_song['track_name'],
            'album': best_song.get('album_name', 'Unknown Album'),
            'artist': best_song.get('artist', 'Unknown Artist'),
            'url': search_url,
            'match_score': 1.0 - best_song['distance'],  # Inverse of distance
            'features': features_found,
            'distance': best_song['distance']
        }

# --- TEST THE FEATURE-BASED BRAIN ---
if __name__ == "__main__":
    dj = OfflineSpotifyBrain()
    
    # SCENARIO 1: DEEP FOCUS (The "Study" Playlist)
    # We want: Low Energy, High Acousticness, Low Liveness (Studio quality), Moderate Loudness
    print("\n🧠 SEARCHING FOR: Deep Focus (Calm & Acoustic)")
    target = {
        'energy': 0.2,
        'valence': 0.5,
        'acousticness': 0.8,  # Acoustic instruments
        'liveness': 0.1,      # Studio recording (no crowd noise)
        'loudness': -10.0     # Quieter, less compressed
    }
    song = dj.get_track_from_features(target)
    if song:
        print(f"Found: {song['title']} (Album: {song['album']})")
        print(f"Match Score: {song['match_score']:.3f} (Distance: {song['distance']:.3f})")
        if song['features']:
            feat_str = ", ".join([f"{k}={v:.3f}" for k, v in song['features'].items()])
            print(f"Features: {feat_str}")
        print(f"Link: {song['url']}")

    # SCENARIO 2: ACOUSTIC SADNESS (The "Breakup" Playlist)
    # We want: Low Valence, Low Energy, High Acousticness
    print("\n😭 SEARCHING FOR: Acoustic Sadness")
    target = {
        'valence': 0.1,       # Very sad
        'energy': 0.2,        # Low energy
        'acousticness': 0.9,  # Very unplugged (guitar/piano)
        'liveness': 0.1,      # Studio quality
        'loudness': -8.0      # Moderate volume
    }
    song = dj.get_track_from_features(target)
    if song:
        print(f"Found: {song['title']} (Album: {song['album']})")
        print(f"Match Score: {song['match_score']:.3f} (Distance: {song['distance']:.3f})")
        if song['features']:
            feat_str = ", ".join([f"{k}={v:.3f}" for k, v in song['features'].items()])
            print(f"Features: {feat_str}")
        print(f"Link: {song['url']}")
    
    # SCENARIO 3: HIGH ENERGY PARTY (The "Workout" Playlist)
    # We want: High Energy, High Valence, Low Acousticness, High Loudness
    print("\n🎉 SEARCHING FOR: High Energy Party Song")
    target = {
        'valence': 0.8,       # Very happy
        'energy': 0.9,        # High energy
        'acousticness': 0.1,  # Electronic/produced
        'liveness': 0.2,      # Studio recording
        'loudness': -3.0      # Loud and punchy
    }
    song = dj.get_track_from_features(target)
    if song:
        print(f"Found: {song['title']} (Album: {song['album']})")
        print(f"Match Score: {song['match_score']:.3f} (Distance: {song['distance']:.3f})")
        if song['features']:
            feat_str = ", ".join([f"{k}={v:.3f}" for k, v in song['features'].items()])
            print(f"Features: {feat_str}")
        print(f"Link: {song['url']}")