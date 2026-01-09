# src/optimizer.py
from bayes_opt import BayesianOptimization #type: ignore
from bayes_opt.acquisition import UpperConfidenceBound #type: ignore
from typing import Optional, Dict, Any #type: ignore
from .backend import SongFinder
from .spotify import SpotifyHandler

class NeuroManager:
    def __init__(self, spotify_id: str = None, spotify_secret: str = None, csv_path: str = "data/neurodj_data.csv"):
        # Initialize the subsystems
        # If ID/Secret are None, they will be loaded from .env by SpotifyHandler
        self.backend = SongFinder(csv_path)
        self.handler = SpotifyHandler(spotify_id, spotify_secret)
        
        # Initialize the Brain (Optimizer)
        # We use UCB (Upper Confidence Bound) to balance exploration vs exploitation
        acquisition = UpperConfidenceBound(kappa=2.5)
        
        self.bo = BayesianOptimization(
            f=None,
            pbounds={
                'valence': (0, 1), 
                'energy': (0, 1)
            },
            acquisition_function=acquisition,
            verbose=0,
            random_state=42,
            allow_duplicate_points=True
        )
        
        self.current_song_data: Optional[Dict[str, Any]] = None 

    def _get_mood_filters(self, mood: str) -> Dict[str, Any]:
        """
        Translates a Brain State (Mood) into Mirrorball Filters.
        This is the "Bridge" between the Brain and the Lyrics Database.
        """
        if not mood:
            return {}
            
        mood = mood.lower()
        filters = {}

        # 1. FOCUS MODE -> Low Complexity
        if mood == 'focus':
            # "Don't distract me with complex poetry"
            # Filters out high lexical diversity songs
            filters['max_complexity'] = 0.45 

        # 2. BOREDOM (Sad/Neutral) -> High Bridge Shift
        elif mood == 'bored' or mood == 'neutral':
            # "Wake me up with a drop"
            # Prioritizes songs with explosive bridges (e.g. Cruel Summer)
            filters['min_bridge_shift'] = 0.7

        # 3. SADNESS -> Avoid "Glitter Gel Pen"
        elif mood == 'sad':
            # "No happy pop songs right now"
            # Explicitly removes the upbeat/frivolous cluster
            filters['exclude_cluster'] = 'Glitter Gel Pen'

        # 4. ANGER -> (Optional) Could lock to Revenge Anthem
        elif mood == 'anger':
            pass

        if filters:
            print(f"Applying Filters for {mood.upper()}: {filters}")
            
        return filters

    def next_song(self, mood: str = None) -> str:
        """
        Main Loop:
        1. Get filters based on current mood
        2. Ask AI for audio targets (Valence/Energy)
        3. Find song matching BOTH audio targets AND lyric filters
        """
        # 1. Determine Filters from Brain State
        filters = self._get_mood_filters(mood)

        # 2. Ask AI for features
        target = self.bo.suggest()
        
        # 3. Get song from Backend (NOW WITH FILTERS)
        song_data = self.backend.get_next_song(target, filters=filters)
        
        if not song_data:
            return "Error: No song found"
            
        self.current_song_data = song_data
        
        # 4. Play it
        success = self.handler.play_specific_song(song_data['name'], song_data['artist'])
        
        if success:
            return song_data['name']
        else:
            return "Error Playing Song"

    def register_feedback(self, score: float, current_mood: str = None) -> Optional[str]:
        """
        User Feedback Loop.
        If they skip, we try again immediately, respecting the current mood.
        
        Args:
            score: 0.0 (Skip) to 1.0 (Like)
            current_mood: The active brain state (so we don't lose the filters on retry)
        """
        if self.current_song_data:
            # Extract only the features that the optimizer expects (valence, energy)
            features = self.current_song_data['features']
            params = {
                'valence': features.get('valence', 0.5),
                'energy': features.get('energy', 0.5)
            }
            
            try:
                self.bo.register(params=params, target=score)
                print(f"Updated Model | Reward: {score}")
            except (KeyError, ValueError) as e:
                # BayesianOpt throws error if point is duplicate; ignore it
                pass

            if score == 0.0:
                print("Skipping...")
                # Pass the mood so the retry uses the correct filters!
                return self.next_song(mood=current_mood)
                
        return None

    def start_with_mood(self, mood: str):
        """
        Bypasses the optimizer to pick the first song based on 
        the initial Brain State. Now applies Filters too.
        """
        print(f"Seeding Engine with Initial Mood: {mood.upper()}")
        
        # Hardcoded 'Centroids' for each mood (Audio Features)
        mood_map = {
            "sad":   {'valence': 0.2, 'energy': 0.2},
            "happy": {'valence': 0.9, 'energy': 0.8},
            "anger": {'valence': 0.1, 'energy': 0.9},
            "focus": {'valence': 0.5, 'energy': 0.3},
            "bored": {'valence': 0.5, 'energy': 0.7}
        }
        
        # Default to 'focus' if mood is unknown
        target_features = mood_map.get(mood.lower(), mood_map['focus'])
        
        # Get Lyric Filters
        filters = self._get_mood_filters(mood)
        
        # Get song from Backend directly
        song_data = self.backend.get_next_song(target_features, filters=filters)
        
        if not song_data:
            return "Error: No song found"
            
        self.current_song_data = song_data
        
        # Play it
        success = self.handler.play_specific_song(song_data['name'], song_data['artist'])
        return song_data['name'] if success else "Error"