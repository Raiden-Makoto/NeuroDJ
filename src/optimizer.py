# src/optimizer.py
from bayes_opt import BayesianOptimization #type: ignore
from bayes_opt.acquisition import UpperConfidenceBound #type: ignore
from typing import Optional, Dict, Any
from .backend import SongFinder
from .spotify import SpotifyHandler

class NeuroManager:
    def __init__(self, spotify_id: str = None, spotify_secret: str = None, csv_path: str = "data/neurodj_data.csv"):
        """
        Initialize the Neuro-DJ manager.
        
        Args:
            spotify_id: Spotify API client ID (optional, loads from .env if not provided)
            spotify_secret: Spotify API client secret (optional, loads from .env if not provided)
            csv_path: Path to the song database CSV file
        """
        # Initialize the subsystems
        self.backend = SongFinder(csv_path)
        self.handler = SpotifyHandler(spotify_id, spotify_secret)
        
        # Initialize the Brain (Optimizer) with valence and energy only
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
        
        self.current_song_data: Optional[Dict[str, Any]] = None  # Stores features of currently playing song

    def start_with_mood(self, mood: str) -> str:
        """
        Bypasses the optimizer to pick the first song based on 
        the initial Brain State.
        """
        print(f"🧠 Seeding Engine with Initial Mood: {mood.upper()}")
        
        # Hardcoded 'Centroids' for each mood
        # This gives the AI a good starting place
        mood_map = {
            "sad":   {'valence': 0.2, 'energy': 0.2},
            "happy": {'valence': 0.9, 'energy': 0.8},
            "anger": {'valence': 0.1, 'energy': 0.9},
            "focus": {'valence': 0.5, 'energy': 0.3}
        }
        
        # Default to 'focus' if mood is unknown
        target_features = mood_map.get(mood, mood_map['happy'])
        
        # Get song from Backend directly
        song_data = self.backend.get_next_song(target_features)
        
        if not song_data:
            return "Error: No song found"
            
        self.current_song_data = song_data
        
        # Play it
        success = self.handler.play_specific_song(song_data['name'], song_data['artist'])
        return song_data['name'] if success else "Error"

    def next_song(self) -> str:
        """
        The Main Loop Step:
        1. AI Suggests parameters
        2. Backend finds matching song
        3. Handler plays it
        
        Returns:
            Song name if successful, error message otherwise
        """
        # 1. Ask AI for features
        target = self.bo.suggest()
        
        # 2. Get song from CSV
        song_data = self.backend.get_next_song(target)
        if not song_data:
            return "Error: No song found"
            
        self.current_song_data = song_data  # Save for later feedback
        
        # 3. Play it on Real Spotify
        success = self.handler.play_specific_song(song_data['name'], song_data['artist'])
        
        if success:
            return song_data['name']
        else:
            return "Error Playing Song"

    def register_feedback(self, score: float) -> Optional[str]:
        """
        Register user feedback and update the model.
        
        Args:
            score: User feedback score
                0.0 = Skip (Bad)
                1.0 = Like (Good)
        
        Returns:
            Next song name if skipped (score == 0.0), None otherwise
        """
        if self.current_song_data:
            # Teach the AI the REAL features of the song we just heard
            # Use only valence and energy from the backend
            all_features = self.current_song_data['features']
            params = {
                'valence': all_features.get('valence', 0.5),
                'energy': all_features.get('energy', 0.5)
            }
            
            try:
                self.bo.register(params=params, target=score)
                print(f"🧠 Updated Model | Reward: {score}")
            except (KeyError, ValueError) as e:
                # Duplicate point or invalid parameters
                print(f"⚠️ Could not register feedback: {e}")

            # If they skipped, immediately find a new song
            if score == 0.0:
                print("⏭️ Skipping...")
                return self.next_song()
                
        return None