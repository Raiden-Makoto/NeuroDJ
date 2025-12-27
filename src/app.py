import streamlit as st
import pandas as pd
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import your actual backend logic
from src.optimizer import NeuroManager
from data.brain import get_multichannel_eeg
from utils.bci_pipe import extract_features
from utils.classifier import classify_mood

# --- CONFIGURATION ---
st.set_page_config(page_title="Neuro-DJ", layout="centered")
load_dotenv()  # Loads SPOTIFY_CLIENT_ID and SECRET from .env

# --- CUSTOM CSS (SPOTIFY DARK MODE) ---
st.markdown("""
<style>
    /* Background & Text */
    .stApp { background-color: #121212; color: white; }
    h1, h2, h3, h4, h5, h6 { color: white; font-family: 'Circular', sans-serif; }
    p, .stMarkdown { color: #b3b3b3; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #1DB954; font-size: 1.2rem; }
    div[data-testid="stMetricLabel"] { color: #b3b3b3; }
    
    /* Buttons */
    .stButton>button {
        background-color: #1DB954;
        color: white !important;
        border-radius: 50px;
        border: none;
        padding: 15px 32px;
        font-weight: bold !important;
        font-size: 16px;
        transition: all 0.2s ease-in-out; 
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button * {
        font-weight: bold !important;
    }
    .stButton>button:hover {
        background-color: #1ed760;
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.4);
    }
    
    /* Album Art Card */
    .album-art {
        border-radius: 8px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'dj' not in st.session_state:
    # Initialize the Manager with credentials from .env
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        st.error("Missing Credentials! Please create a .env file with SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.")
        st.stop()
        
    with st.spinner("Booting Neuro-DJ..."):
        try:
            # NeuroManager now loads credentials from .env automatically
            st.session_state.dj = NeuroManager()
            st.success("Connected to Spotify & Brain Backend!")
            time.sleep(1) # Show success briefly
            st.rerun()
        except Exception as e:
            st.error(f"Failed to connect: {e}")
            st.stop()

if 'history' not in st.session_state:
    st.session_state.history = []

if 'last_track_id' not in st.session_state:
    st.session_state.last_track_id = None

if 'playback_toggled' not in st.session_state:
    st.session_state.playback_toggled = False

if 'session_started' not in st.session_state:
    st.session_state.session_started = False

if 'current_brain_state' not in st.session_state:
    st.session_state.current_brain_state = None

if 'pending_mood_change' not in st.session_state:
    st.session_state.pending_mood_change = None

if 'auto_liked_tracks' not in st.session_state:
    st.session_state.auto_liked_tracks = set()  # Track IDs that have been auto-liked

if 'next_song_queued' not in st.session_state:
    st.session_state.next_song_queued = False  # Track if next song is already queued

if 'queued_song_data' not in st.session_state:
    st.session_state.queued_song_data = None  # Store next song info for display

if 'auto_like_pending_rerun' not in st.session_state:
    st.session_state.auto_like_pending_rerun = False  # Flag to trigger rerun after auto-like

if 'song_start_time' not in st.session_state:
    st.session_state.song_start_time = None  # Track when current song started playing

if 'last_periodic_rerun_time' not in st.session_state:
    st.session_state.last_periodic_rerun_time = 0  # Track last periodic rerun for auto-like checking

if 'should_play_queued_song' not in st.session_state:
    st.session_state.should_play_queued_song = False  # Flag to ensure queued song plays when current ends

if 'skip_pending_rerun' not in st.session_state:
    st.session_state.skip_pending_rerun = False  # Flag to trigger rerun after skip

if 'next_in_queue_pending_rerun' not in st.session_state:
    st.session_state.next_in_queue_pending_rerun = False  # Flag to trigger rerun after playing queued song

if 'quit_session_pending_rerun' not in st.session_state:
    st.session_state.quit_session_pending_rerun = False  # Flag to trigger rerun after quitting session

if 'track_changed_pending_rerun' not in st.session_state:
    st.session_state.track_changed_pending_rerun = False  # Flag to trigger rerun when track changes


if 'song_played_pending_rerun' not in st.session_state:
    st.session_state.song_played_pending_rerun = False  # Flag to trigger rerun after playing a new song

if 'periodic_rerun_pending' not in st.session_state:
    st.session_state.periodic_rerun_pending = False  # Flag to trigger periodic rerun at end of script

if 'last_sim_mood_selection' not in st.session_state:
    st.session_state.last_sim_mood_selection = None  # Track last sidebar mood selection

# --- HELPER: SYNC WITH SPOTIFY ---
def get_current_spotify_state():
    """
    Polls the real Spotify app to get accurate Album Art and Progress.
    Returns None if no playback or error occurs.
    """
    sp = st.session_state.dj.handler.sp
    try:
        current = sp.current_playback()
        if current and current.get('item'):
            track = current['item']
            track_id = track.get('id')  # Unique identifier for the track
            if not track_id:
                return None
            
            progress_ms = current.get('progress_ms', 0)
            current_time = time.time()
            
            # Check if song changed - set flags for main script to handle
            if st.session_state.last_track_id != track_id:
                # If we have a queued song and track changed, set flag to play it
                if st.session_state.next_song_queued and st.session_state.queued_song_data:
                    st.session_state.should_play_queued_song = True
                
                st.session_state.last_track_id = track_id
                # Reset song start time when new song starts
                st.session_state.song_start_time = current_time
                # Set flag to trigger rerun (handled in main script)
                st.session_state.track_changed_pending_rerun = True
            
            # Track song start time for reference (auto-like now uses progress_ms directly)
            if st.session_state.song_start_time is None or st.session_state.last_track_id != track_id:
                st.session_state.song_start_time = current_time
            
            # Safely extract track information with fallbacks
            artists = track.get('artists', [])
            artist_name = artists[0]['name'] if artists else 'Unknown Artist'
            
            album = track.get('album', {})
            album_name = album.get('name', 'Unknown Album')
            
            images = album.get('images', [])
            cover_url = images[0]['url'] if images else 'https://via.placeholder.com/300'
            
            return {
                "title": track.get('name', 'Unknown Track'),
                "artist": artist_name,
                "album": album_name,
                "cover": cover_url,
                "progress_ms": progress_ms,
                "duration_ms": track.get('duration_ms', 0),
                "is_playing": current.get('is_playing', False),
                "track_id": track_id
            }
    except Exception as e:
        # Log error but don't crash
        print(f"Error fetching Spotify state: {e}")
        return None
    
    # Fallback if nothing is playing
    return None

# --- ACTION FUNCTIONS ---
def handle_skip():
    """Tell AI we hated it -> Play new song -> Update UI"""
    with st.spinner("Skipping... AI is learning..."):
        current = get_current_spotify_state()
        if current:
            # Check if this track_id is already in history (duplicate)
            track_ids_in_history = [h.get('track_id') for h in st.session_state.history if 'track_id' in h]
            is_duplicate = current.get('track_id') in track_ids_in_history
            
            # Only submit feedback if NOT a duplicate
            if not is_duplicate:
                # 0.0 Reward = Skip
                new_song = st.session_state.dj.register_feedback(0.0)
            else:
                # Duplicate song - just skip without feedback
                new_song = st.session_state.dj.next_song()
            
            # Log to history (prevent duplicates)
            if not is_duplicate:
                st.session_state.history.insert(0, {
                    "Track": current['title'],
                    "Artist": current['artist'],
                    "Reaction": "Skipped",
                    "track_id": current.get('track_id')
                })
            
            # Clear the queue when skipping
            st.session_state.next_song_queued = False
            st.session_state.queued_song_data = None
        
        # Set flag to trigger rerun after skip completes
        st.session_state.skip_pending_rerun = True

def handle_next_in_queue():
    """Play the next song in queue and clear the queue"""
    if st.session_state.next_song_queued and st.session_state.queued_song_data:
        queued_song = st.session_state.queued_song_data
        
        # Play the queued song
        success = st.session_state.dj.handler.play_specific_song(
            queued_song['name'], 
            queued_song['artist']
        )
        
        if success:
            # Update current song data
            st.session_state.dj.current_song_data = queued_song
            st.toast(f"Playing: {queued_song['name']}")
        else:
            st.error("Failed to play queued song")
        
        # Clear the queue
        st.session_state.next_song_queued = False
        st.session_state.queued_song_data = None
        
        # Set flag to trigger rerun
        st.session_state.next_in_queue_pending_rerun = True
    else:
        st.warning("No song in queue")

def handle_play_pause():
    """Toggle Playback directly on Spotify"""
    handler = st.session_state.dj.handler
    try:
        # Get current state to check if playing
        current = handler.sp.current_playback()
        is_playing = current and current.get('is_playing', False) if current else False
        
        if is_playing:
            # Currently playing - pause it
            handler.sp.pause_playback()
            st.toast("Paused")
        else:
            # Not playing - start/resume playback
            handler.sp.start_playback()
            st.toast("Playing")
        # Set flag to trigger refresh
        st.session_state.playback_toggled = True
    except Exception as e:
        error_msg = str(e)
        if "NO_ACTIVE_DEVICE" in error_msg or "404" in error_msg:
            st.warning("No active device. Please open Spotify and start playing a song first.")
        else:
            st.warning(f"Control Error: {e}")

def handle_quit_session():
    """End the session, reset state, and clear cache (but keep music playing)"""
    # Clear all session state
    st.session_state.session_started = False
    st.session_state.current_brain_state = None
    st.session_state.pending_mood_change = None
    st.session_state.history = []
    st.session_state.last_track_id = None
    st.session_state.auto_liked_tracks = set()
    st.session_state.next_song_queued = False
    st.session_state.queued_song_data = None
    st.session_state.playback_toggled = False
    
    # Clear Streamlit cache
    st.cache_data.clear()
    
    st.toast("Session ended - Music continues playing")
    # Set flag to trigger rerun
    st.session_state.quit_session_pending_rerun = True

# --- UI LAYOUT ---

# Add a Sidebar Debugger so you can control the simulation
with st.sidebar:
    st.header("Hardware Simulator")
    st.selectbox(
        "Force Brain State:", 
        ['sad', 'happy', 'anger', 'focus'], 
        key='sim_mood_selection'
    )

# 1. HEADER & BRAIN SENSOR
c1, c2 = st.columns([3, 1])
with c1:
    st.title("Neuro-DJ")
    st.caption("Bayesian Optimization Music Interface")

with c2:
    # Show different button based on session state
    if st.session_state.session_started:
        if st.button("QUIT SESSION", type="secondary"):
            handle_quit_session()
    else:
        # THE START BUTTON NOW READS THE BRAIN FIRST
        if st.button("SCAN & START"):
            with st.spinner("Reading EEG Signals..."):
                # 1. Hardware Simulation
                # In a real app, this connects to the headset stream
                # For now, we simulate a random state or let you pick in sidebar
                sim_state = st.session_state.get('sim_mood_selection', 'focus')
                raw_eeg, fs = get_multichannel_eeg(mood=sim_state)
                
                # 2. Signal Processing
                features = extract_features(raw_eeg, fs)
                detected_mood = classify_mood(features)
                
                st.toast(f"Detected Brain State: {detected_mood.upper()}")
                time.sleep(1) # Dramatic pause
                
                # 3. The Cold Start
                st.session_state.dj.start_with_mood(detected_mood)
                st.session_state.session_started = True
                st.session_state.current_brain_state = detected_mood
                st.rerun()

st.divider()

# 2. MAIN PLAYER CARD
# Check if playback was toggled and refresh if needed (st.rerun() doesn't work in callbacks)
if st.session_state.get('playback_toggled', False):
    st.session_state.playback_toggled = False
    st.rerun()

# Check if auto-like needs to trigger rerun (after delay)
if st.session_state.get('auto_like_pending_rerun', False):
    st.session_state.auto_like_pending_rerun = False
    time.sleep(2.0)
    st.rerun()

# Check if skip needs to trigger rerun
if st.session_state.get('skip_pending_rerun', False):
    st.session_state.skip_pending_rerun = False
    time.sleep(2.0)
    st.rerun()

# Check if next in queue needs to trigger rerun
if st.session_state.get('next_in_queue_pending_rerun', False):
    st.session_state.next_in_queue_pending_rerun = False
    time.sleep(0.5)
    st.rerun()

# Check if quit session needs to trigger rerun
if st.session_state.get('quit_session_pending_rerun', False):
    st.session_state.quit_session_pending_rerun = False
    st.rerun()

# Check if track changed and needs rerun
if st.session_state.get('track_changed_pending_rerun', False):
    st.session_state.track_changed_pending_rerun = False
    st.rerun()

# Check if song was played and needs rerun (after delay)
if st.session_state.get('song_played_pending_rerun', False):
    st.session_state.song_played_pending_rerun = False
    time.sleep(2.0)
    # Verify song is playing before rerun
    verify_state = get_current_spotify_state()
    if verify_state and verify_state.get('is_playing'):
        st.rerun()

# Monitor brain state changes if session is active
# Only check if sidebar mood selection changed - don't rescan brain state on every rerun
if st.session_state.session_started:
    sim_state = st.session_state.get('sim_mood_selection', 'focus')
    
    # Only scan and update brain state if sidebar selection changed
    # This prevents constant rescanning and resetting
    last_sim_state = st.session_state.get('last_sim_mood_selection', None)
    
    if sim_state != last_sim_state:
        # Sidebar mood selection changed - scan brain state
        st.session_state.last_sim_mood_selection = sim_state
        raw_eeg, fs = get_multichannel_eeg(mood=sim_state, duration_sec=5)
        features = extract_features(raw_eeg, fs)
        detected_mood = classify_mood(features)
        
        # Only update if brain state actually changed (preserves state between reruns)
        if st.session_state.current_brain_state != detected_mood:
            if st.session_state.current_brain_state is not None:
                # Brain state changed - queue it to load after current song finishes
                st.session_state.pending_mood_change = detected_mood
                st.toast(f"Brain State Changed: {detected_mood.upper()} (Will load after current song)")
            # Update brain state only when it changes
            st.session_state.current_brain_state = detected_mood
    # If sidebar selection didn't change, brain state is preserved (not reset)

# Fetch real-time state from Spotify FIRST (before any rerun checks)
state = get_current_spotify_state()

# Auto-Like: Check if user has listened for 30+ seconds
# Check directly using progress_ms
if state and state.get('track_id'):
    track_id = state['track_id']
    progress_seconds = state['progress_ms'] / 1000.0
    
    # Check if they've listened for more than 30 seconds and haven't auto-liked yet
    # Only check if song is playing (don't auto-like paused songs)
    if (progress_seconds >= 30.0 and 
        track_id not in st.session_state.auto_liked_tracks and
        state.get('is_playing', False)):
        # Auto-like this track
        st.session_state.auto_liked_tracks.add(track_id)
        
        # Register feedback with the optimizer
        if st.session_state.dj.current_song_data:
            st.session_state.dj.register_feedback(1.0)
        
        # Add to history (prevent duplicates)
        track_ids_in_history = [h.get('track_id') for h in st.session_state.history if 'track_id' in h]
        if track_id not in track_ids_in_history:
            st.session_state.history.insert(0, {
                "Track": state['title'],
                "Artist": state['artist'],
                "Reaction": "Auto-Liked (30s+)",
                "track_id": track_id
            })
        
        st.toast("Auto-liked! (Listened 30+ seconds)")
        
        # Queue next song immediately after auto-like - preview it
        target = st.session_state.dj.bo.suggest()
        queued_song = st.session_state.dj.backend.get_next_song(target)
        st.session_state.queued_song_data = queued_song
        st.session_state.next_song_queued = True
        # Set flag to trigger rerun after delay
        st.session_state.auto_like_pending_rerun = True

# Check song end BEFORE rendering UI - this ensures queued songs play immediately
# If current progress + 10 seconds exceeds song duration, song is finished
if state and state.get('duration_ms', 0) > 0:
    progress_plus_10s = state['progress_ms'] + 10000  # Add 10 seconds (10000 ms)
    song_finished = progress_plus_10s >= state['duration_ms']
    
    # Priority 1: If we have a queued song and song finished, play it immediately
    if song_finished and st.session_state.next_song_queued and st.session_state.queued_song_data:
        queued_song = st.session_state.queued_song_data
        # Play the queued song directly
        success = st.session_state.dj.handler.play_specific_song(
            queued_song['name'], 
            queued_song['artist']
        )
        if success:
            # Update current song data
            st.session_state.dj.current_song_data = queued_song
            st.toast(f"Playing queued song: {queued_song['name']}")
        # Clear the queue
        st.session_state.next_song_queued = False
        st.session_state.queued_song_data = None
        st.session_state.should_play_queued_song = False
        # Wait and rerun
        time.sleep(2.0)
        st.rerun()
    # Priority 2: If we have a pending mood change and song finished, use that
    elif song_finished and st.session_state.pending_mood_change:
        new_mood = st.session_state.pending_mood_change
        st.session_state.pending_mood_change = None
        song_name = st.session_state.dj.start_with_mood(new_mood)
        st.toast(f"Loading song for {new_mood.upper()} mood")
        st.session_state.next_song_queued = False
        st.session_state.queued_song_data = None
        st.session_state.should_play_queued_song = False
        # Wait and rerun
        time.sleep(2.0)
        st.rerun()
    # Priority 3: If song finished naturally (no queue, no mood change), play next song
    elif song_finished:
        song_name = st.session_state.dj.next_song()
        st.session_state.queued_song_data = None
        st.session_state.should_play_queued_song = False
        st.toast("Loading next song")
        # Wait and rerun
        time.sleep(2.0)
        st.rerun()

if state:
    col_art, col_info = st.columns([1, 2])
    
    with col_art:
        # Display Real Album Art
        st.image(state['cover'], width=250, output_format="JPEG")
    
    with col_info:
        st.markdown(f"## {state['title']}")
        st.markdown(f"### {state['artist']}")
        st.markdown(f"*{state['album']}*")
        
        # Progress Bar (with safety check - updates automatically on rerun)
        if state['duration_ms'] > 0:
            progress = state['progress_ms'] / state['duration_ms']
            st.progress(min(progress, 1.0))  # Updates automatically when value changes on rerun
        else:
            st.progress(0.0)
        
        # Time Display
        def fmt_time(ms):
            seconds = int((ms / 1000) % 60)
            minutes = int((ms / (1000 * 60)) % 60)
            return f"{minutes}:{seconds:02d}"
            
        st.caption(f"{fmt_time(state['progress_ms'])} / {fmt_time(state['duration_ms'])}")

        # AI Stats (Hidden Feature)
        with st.expander("View Neural Stats"):
            # Show what the AI *thought* it was playing vs reality
            if st.session_state.dj.current_song_data:
                feats = st.session_state.dj.current_song_data['features']
                c1, c2, c3 = st.columns(3)
                c1.metric("Valence", f"{feats['valence']:.2f}")
                c2.metric("Energy", f"{feats['energy']:.2f}")
                c3.metric("Acoustic", f"{feats['acousticness']:.2f}")

    # 3. CONTROLS
    st.write("")
    b1, b2, b3 = st.columns([1, 1, 1])
    
    with b1:
        st.button("SKIP", on_click=handle_skip, width='stretch')
    
    with b2:
        icon = "PAUSE" if state['is_playing'] else "PLAY"
        st.button(icon, on_click=handle_play_pause, width='stretch')
        
    with b3:
        # Only enable button if there's a song in queue
        button_disabled = not (st.session_state.next_song_queued and st.session_state.queued_song_data)
        st.button("NEXT IN QUEUE", on_click=handle_next_in_queue, width='stretch', disabled=button_disabled)

    # 4. QUEUE DISPLAY
    if st.session_state.next_song_queued and st.session_state.queued_song_data:
        st.divider()
        st.subheader("Next in Queue")
        queue_data = st.session_state.queued_song_data
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**{queue_data['name']}**")
            st.caption(f"by {queue_data['artist']}")
        with col2:
            queue_feats = queue_data['features']
            q1, q2, q3 = st.columns(3)
            q1.metric("Valence", f"{queue_feats['valence']:.2f}")
            q2.metric("Energy", f"{queue_feats['energy']:.2f}")
            q3.metric("Acoustic", f"{queue_feats['acousticness']:.2f}")

else:
    # Empty State
    st.info("Nothing playing. Click 'SCAN & START' to begin your Neuro-DJ session.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/1024px-Spotify_logo_without_text.svg.png", width=150)

# 5. HISTORY
st.divider()
st.subheader("Session History")
if st.session_state.history:
    # Remove track_id from display (it's just for deduplication)
    display_history = [{k: v for k, v in item.items() if k != 'track_id'} for item in st.session_state.history]
    st.dataframe(
        pd.DataFrame(display_history), 
        width='stretch', 
        hide_index=True
    )

# Periodic rerun to ensure get_current_spotify_state() is called regularly
# This allows progress bar updates AND auto-like checks to work
# MUST happen at END after all UI is rendered
# IMPORTANT: Script must keep itself alive - wait 10 seconds then rerun
if st.session_state.session_started:
    # Wait 10 seconds, then rerun to update progress bar and check auto-like
    # This keeps the script alive and ensures continuous updates
    time.sleep(10.0)
    st.rerun()
