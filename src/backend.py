import pandas as pd #type: ignore
import numpy as np #type: ignore

class SongFinder:
    def __init__(self, csv_path="data/neurodj_data.csv"):
        """
        Initialize the SongFinder with the enhanced dataset (Audio + Lyrics).
        """
        try:
            self.df = pd.read_csv(csv_path)
            # Normalize columns to lowercase to match our logic
            self.df.columns = [c.lower() for c in self.df.columns]
        except FileNotFoundError:
            print(f"Warning: Dataset '{csv_path}' not found.")
            self.df = pd.DataFrame()
        except Exception as e:
            print(f"Error loading database: {e}")
            self.df = pd.DataFrame()
            
        self.taboo_list = set()

    def get_next_song(self, target_features, filters=None):
        """
        Find the best song matching the target audio features and optional lyrical filters.
        
        Args:
            target_features: Dict of audio targets (e.g., {'valence': 0.5, 'energy': 0.8})
            filters: Optional Dict of constraints (e.g., {'max_complexity': 0.4})
        """
        # 1. Filter out played songs (Taboo List)
        candidates = self.df[~self.df['track_name'].isin(self.taboo_list)].copy()
        
        # If we've played everything, reset the list
        if candidates.empty:
            print("Resetting Library (All songs played)")
            self.taboo_list.clear()
            candidates = self.df.copy()

        # 2. APPLY MIRRORBALL FILTERS (The New Logic)
        if filters:
            # A. Complexity Filter (For Focus Mode)
            # "Don't play songs with complex lyrics that distract the user"
            if 'max_complexity' in filters and 'lexical_diversity' in candidates.columns:
                val = filters['max_complexity']
                # Keep songs simpler than X OR where data is missing
                candidates = candidates[
                    (candidates['lexical_diversity'] <= val) | 
                    (candidates['lexical_diversity'].isna())
                ]

            # B. Bridge Shift Filter (For Boredom Rescue)
            # "Find a song with a massive drop/shift in the bridge"
            if 'min_bridge_shift' in filters and 'bridge_shift' in candidates.columns:
                val = filters['min_bridge_shift']
                candidates = candidates[
                    (candidates['bridge_shift'] >= val) |
                    (candidates['bridge_shift'].isna())
                ]

            # C. Cluster Exclusions (Mood Guardrails)
            # "Don't play 'Glitter Gel Pen' songs when I am sad"
            if 'exclude_cluster' in filters and 'archetype_name' in candidates.columns:
                cluster = filters['exclude_cluster']
                candidates = candidates[candidates['archetype_name'] != cluster]

        # 3. Fallback Mechanism
        # If filters were too strict and killed all candidates, ignore filters
        if candidates.empty:
            print("Filters too strict! Relaxing them to find a song...")
            candidates = self.df[~self.df['track_name'].isin(self.taboo_list)].copy()

        # 4. Calculate Euclidean Distance (Audio Features)
        candidates['distance'] = 0.0
        for feature, value in target_features.items():
            if feature in candidates.columns:
                # Calculate squared difference
                candidates['distance'] += (candidates[feature] - value) ** 2
        
        # Final Score
        candidates['distance'] = np.sqrt(candidates['distance'])
        
        # 5. Pick the Winner
        try:
            best_row = candidates.loc[candidates['distance'].idxmin()]
        except ValueError:
            # Emergency fallback if something really weird happens
            return None
        
        # 6. Update Taboo List
        self.taboo_list.add(best_row['track_name'])
        
        # 7. Return clean data object
        return {
            "name": str(best_row.get('track_name', 'Unknown Track')),
            "artist": "Taylor Swift",
            "album": str(best_row.get('album_name', 'Unknown Album')),
            "features": {
                "valence": float(best_row.get('valence', 0.5)),
                "energy": float(best_row.get('energy', 0.5))
            },
            # Return new metadata for debugging/UI
            "metadata": {
                "cluster": best_row.get('archetype_name', 'Unknown'),
                "complexity": float(best_row.get('lexical_diversity', 0)),
                "bridge_shift": float(best_row.get('bridge_shift', 0))
            }
        }