# src/spotify.py
import os
import spotipy #type: ignore
from spotipy.oauth2 import SpotifyOAuth #type: ignore
from dotenv import load_dotenv #type: ignore

# Load environment variables
load_dotenv()

class SpotifyHandler:
    def __init__(self, client_id=None, client_secret=None):
        """
        Initialize Spotify handler with OAuth authentication.
        Loads credentials from .env file if not provided.
        
        Args:
            client_id: Spotify API client ID (optional, loads from .env if not provided)
            client_secret: Spotify API client secret (optional, loads from .env if not provided)
        """
        # Load from .env if not provided
        if not client_id:
            client_id = os.getenv('SPOTIFY_CLIENT_ID')
        if not client_secret:
            client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        # Validate credentials
        if not client_id or not client_secret:
            raise ValueError(
                "Spotify client_id and client_secret must be provided either as "
                "parameters or in .env file as SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET"
            )
        
        # Strip whitespace that might cause issues
        client_id = client_id.strip()
        client_secret = client_secret.strip()
        
        # We need 'user-modify-playback-state' to control the player
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri="http://127.0.0.1:6767",
                scope="user-modify-playback-state user-read-playback-state"
            ))
        except Exception as e:
            raise ValueError(f"Failed to initialize Spotify authentication: {e}. Check your client_id and client_secret.")

    def play_specific_song(self, song_name, artist_name):
        """
        Input: "Style", "Taylor Swift"
        Action: Searches Spotify -> Finds URI -> Starts Playback
        """
        # 1. Search for the specific track
        query = f"track:{song_name} artist:{artist_name}"
        results = self.sp.search(q=query, type='track', limit=1)
        
        tracks = results['tracks']['items']
        if not tracks:
            print(f"❌ Spotify could not find: {song_name}")
            return False

        # 2. Get the URI (The ID code)
        track_uri = tracks[0]['uri']
        print(f'Found song: {track_uri}')
        
        # 3. Get available devices and select one
        try:
            devices = self.sp.devices()
            available_devices = devices.get('devices', [])
            
            # Find active device first
            active_device = None
            for device in available_devices:
                if device.get('is_active', False):
                    active_device = device
                    break
            
            # If no active device, use the first available device
            if not active_device and available_devices:
                active_device = available_devices[0]
                print(f"📱 Using device: {active_device['name']}")
            
            device_id = active_device['id'] if active_device else None
            
        except Exception as e:
            print(f"⚠️ Could not get devices: {e}")
            device_id = None
        
        # 4. Send Play Command
        try:
            if device_id:
                self.sp.start_playback(device_id=device_id, uris=[track_uri])
            else:
                self.sp.start_playback(uris=[track_uri])
            print(f"▶️ Now Playing: {song_name}")
            return True
        except Exception as e:
            error_msg = str(e)
            if "NO_ACTIVE_DEVICE" in error_msg or "404" in error_msg:
                print(f"⚠️ No active device. Please:")
                print(f"   1. Open Spotify app")
                print(f"   2. Start playing any song (or select a device)")
                print(f"   3. Try again")
            else:
                print(f"⚠️ Playback Error: {e}")
            return False