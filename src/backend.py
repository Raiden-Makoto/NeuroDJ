# src/backend.py
import pandas as pd #type: ignore
import numpy as np #type: ignore
from typing import Dict, Any

class SongFinder:
    def __init__(self, csv_path="data/taylor_swift.csv"):
        self.df = pd.read_csv(csv_path)
        # Normalize columns
        self.df.columns = [c.lower() for c in self.df.columns]
        self.taboo_list = set()

    def get_next_song(self, target_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Find the closest matching song based on target features.
        
        Args:
            target_features: Dictionary of feature names to target values
                e.g., {'valence': 0.5, 'energy': 0.8}
        
        Returns:
            Dictionary with song information:
            {
                'name': str,
                'artist': str,
                'album': str,
                'features': {
                    'valence': float,
                    'energy': float
                }
            }
        """
        # 1. Filter out played songs
        candidates = self.df[~self.df['track_name'].isin(self.taboo_list)].copy()
        
        if candidates.empty:
            print("⚠️ Resetting Library (All songs played)")
            self.taboo_list.clear()
            candidates = self.df.copy()

        # 2. Calculate Euclidean Distance
        candidates['distance'] = 0.0
        for feature, value in target_features.items():
            if feature in candidates.columns:
                candidates['distance'] += (candidates[feature] - value) ** 2
        
        candidates['distance'] = np.sqrt(candidates['distance'])
        
        # 3. Pick Winner
        best_row = candidates.loc[candidates['distance'].idxmin()]
        
        # 4. Update Taboo List
        self.taboo_list.add(best_row['track_name'])
        
        # 5. Return pure data with safe type conversions
        try:
            return {
                "name": str(best_row.get('track_name', 'Unknown Track')),
                "artist": str(best_row.get('artist', 'Taylor Swift')),  # Use artist from CSV, fallback to Taylor Swift
                "album": str(best_row.get('album_name', 'Unknown Album')),
                "features": {
                    "valence": float(best_row.get('valence', 0.5)),
                    "energy": float(best_row.get('energy', 0.5))
                }
            }
        except (ValueError, KeyError) as e:
            print(f"Error processing song data: {e}")
            # Return a safe default
            return {
                "name": "Unknown Track",
                "artist": "Unknown Artist",
                "album": "Unknown Album",
                "features": {
                    "valence": 0.5,
                    "energy": 0.5
                }
            }