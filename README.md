![NeuroDJ Logo](neurodj.png)

## Overview

NeuroDJ combines neuroscience-inspired signal processing with machine learning to create an adaptive music recommendation system. The system simulates EEG (electroencephalogram) brainwave patterns based on your mood, then uses Bayesian optimization to learn your music preferences through interactive feedback.

**Key Features:**
- 🧠 **Brain State Detection**: Simulates EEG signals to detect your emotional state (sad, happy, anger, focus)
- 🎵 **Spotify Integration**: Directly controls your Spotify playback
- 🤖 **Auto-Like**: Automatically likes songs you listen to for 30+ seconds
- 📊 **Bayesian Optimization**: Learns your music preferences through feedback
- 🎯 **Smart Queueing**: Previews and queues the next song based on your mood
- 📈 **Real-time Progress**: Updates every 10 seconds automatically

---

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.8+** installed on your system
2. **A Spotify Premium account** (required for playback control)
3. **Spotify Desktop App or Web Player** installed and logged in
4. **Spotify Developer Account** (free) to get API credentials

---

## TO RUN NEURODJ ON YOUR DEVICE

### 1. Clone or Download the Repository

```bash
cd /path/to/NeuroDJ
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `numpy`, `pandas`, `scipy`, `matplotlib` - Data processing and visualization
- `bayesian-optimization` - Machine learning optimization
- `spotipy==2.25.2` - Spotify API wrapper
- `streamlit==1.52` - Web interface framework
- `watchdog` - File system monitoring
- `python-dotenv` - Environment variable management

### 3. Set Up Spotify API Credentials

#### Step 1: Create a Spotify App

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Log in with your Spotify account
3. Click **"Create App"**
4. Fill in:
   - **App Name**: `My Music App` (or any name you prefer - must be different from "NeuroDJ")
   - **App Description**: `Music recommendation system`
   - **Redirect URI**: `http://127.0.0.1:XXXX` (replace XXXX with the port number - **check `src/spotify.py` for the actual port used in the code**)
   - Check the checkbox to agree to terms
5. Click **"Save"**

**Important**: The redirect URI port must match the port configured in the code. Check `src/spotify.py` line 42 to see the actual port number (currently `6767`). Use that same port in your Spotify app settings.

#### Step 2: Get Your Credentials

1. In your app dashboard, click **"Settings"**
2. Copy your **Client ID** and **Client Secret**
3. Click **"Edit Settings"** and add `http://127.0.0.1:XXXX` to **Redirect URIs** (use the same port from `src/spotify.py`)
4. Click **"Add"** and **"Save"**

#### Step 3: Create `.env` File

Create a file named `.env` in the project root directory:

```bash
touch .env
```

Add your credentials to the `.env` file:

```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

**Important**: Never commit the `.env` file to version control! It contains sensitive credentials.

---

## Download Music Dataset

**⚠️ IMPORTANT**: Spotify removed audio features (valence, energy, etc.) from their public API. You must download a dataset manually.

### Option 1: Use Provided Dataset

The project includes a sample dataset (`data/taylor_swift.csv`) that you can use to test the application.

### Option 2: Download Your Own Dataset

You need a CSV file with the following columns:
- `album_name` - Name of the album
- `track_name` - Name of the track
- `artist` - Artist name
- `valence` - Musical positiveness (0.0 to 1.0)
- `energy` - Perceived energy (0.0 to 1.0)

**Where to get datasets:**
- [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
- [Kaggle Spotify Datasets](https://www.kaggle.com/datasets?search=spotify)
- Use Spotify's Web API with a developer account to extract features (requires additional setup)

**Place your dataset:**
1. Save your CSV file in the `data/` directory
2. Name it `taylor_swift.csv` (or update the path in `src/backend.py` and `src/optimizer.py`)

**Example CSV format:**
```csv
album_name,track_name,artist,valence,energy
Album Name,Song Title,Artist Name,0.5,0.7
```

---

## Running the Application

### Start the Streamlit App

```bash
streamlit run src/app.py
```

The app will:
1. Open in your default web browser at `http://localhost:8501`
2. Attempt to connect to Spotify (you may need to authorize on first run)
3. Display the NeuroDJ interface

### First-Time Authorization

On first run, you'll be redirected to Spotify's authorization page:
1. Click **"Agree"** to authorize NeuroDJ
2. You'll be redirected back to the app
3. The app is now connected to your Spotify account

---

## How to Use

### Starting a Session

1. **Open Spotify** on your computer (Desktop app or web player)
2. **Start playing any song** in Spotify (this activates your device)
3. In the NeuroDJ interface, use the sidebar to select your current mood:
   - `sad` - For melancholic, introspective music
   - `happy` - For upbeat, energetic music
   - `anger` - For intense, powerful music
   - `focus` - For calm, concentration-friendly music
4. Click **"SCAN & START"** button
5. The app will:
   - Simulate reading your brain state
   - Detect your mood
   - Start playing a song that matches your mood

### During Playback

#### Main Controls

- **SKIP** - Skip the current song and tell the AI you didn't like it (registers negative feedback)
- **PAUSE/PLAY** - Toggle playback on Spotify
- **NEXT IN QUEUE** - Play the queued song immediately (only enabled when a song is queued)

#### Auto-Like Feature

- Songs are **automatically liked** if you listen for **30+ seconds**
- This registers positive feedback with the AI
- A toast notification will appear: "Auto-liked! (Listened 30+ seconds)"
- The next song will be queued automatically after auto-like

#### Progress Bar

- Updates **every 10 seconds** automatically
- Shows current progress through the song
- Displays time remaining

#### Song Queue

- After auto-liking a song, the next song is automatically queued
- You can see the queued song's details (name, artist, features)
- The queued song will play automatically when the current song ends
- Or click **"NEXT IN QUEUE"** to play it immediately

### Changing Your Mood

1. Use the sidebar **"Force Brain State"** dropdown
2. Select a new mood (sad, happy, anger, focus)
3. The brain state will be re-scanned
4. A new song matching the new mood will be queued
5. The new song will play **after the current song finishes**

### Ending a Session

1. Click **"QUIT SESSION"** button
2. This will:
   - End the NeuroDJ session
   - Clear the history and queue
   - **Keep your music playing** (doesn't stop Spotify)
3. You can start a new session anytime

---

## Understanding the Features

### Brain State Detection

- Simulates EEG (electroencephalogram) signals based on your selected mood
- Extracts features from the simulated brainwave patterns
- Classifies your emotional state using machine learning
- Influences song selection to match your mood

### Bayesian Optimization

- Learns your music preferences through feedback:
  - **Positive feedback**: Auto-like (30+ seconds) or manual likes
  - **Negative feedback**: Skipping songs
- Uses Bayesian optimization to suggest songs you'll enjoy
- Gets smarter over time as you use it more

### Song Selection Priority

When a song ends, the next song is selected in this order:

1. **Queued Song** - If a song is already queued (from auto-like or mood change)
2. **Mood Change** - If you changed your mood, a song matching the new mood
3. **Natural Next** - The next song suggested by the optimizer

### Song End Detection

- Songs switch automatically when there are **less than 10 seconds remaining**
- Uses the formula: `(current progress + 10 seconds) >= song duration`
- Ensures songs play almost to completion

### Session History

- Tracks all songs played during the session
- Shows your reaction (Skipped, Auto-Liked, etc.)
- Displays in a table at the bottom of the interface

---

## Troubleshooting

### "Missing Credentials" Error

**Problem**: App shows "Missing Credentials! Please create a .env file..."

**Solution**:
1. Ensure you created a `.env` file in the project root
2. Check that it contains:
   ```
   SPOTIFY_CLIENT_ID=your_actual_client_id
   SPOTIFY_CLIENT_SECRET=your_actual_client_secret
   ```
3. Make sure there are no extra spaces or quotes around the values
4. Restart the Streamlit app

### "No active device" Error

**Problem**: Controls don't work, shows "No active device"

**Solution**:
1. **Open Spotify** on your computer (Desktop app or web player)
2. **Start playing any song** in Spotify
3. This activates your device for the API
4. Try the controls again

### Authorization Issues

**Problem**: Can't connect to Spotify, authorization fails

**Solution**:
1. Check that your Redirect URI in Spotify Dashboard matches the port in `src/spotify.py` (check line 42 for the actual port number)
2. Clear your browser cache and cookies
3. Try authorizing again
4. Make sure you're using a **Spotify Premium** account

### Progress Bar Not Updating

**Problem**: Progress bar stays frozen

**Solution**:
1. The app updates every 10 seconds automatically
2. If it's not updating, check that:
   - A song is actually playing in Spotify
   - Your internet connection is working
   - The Streamlit app is still running
3. Try refreshing the browser page

### Auto-Like Not Working

**Problem**: Songs aren't being auto-liked after 30 seconds

**Solution**:
1. Make sure the song is **actually playing** (not paused)
2. Wait for the 10-second update cycle
3. Check that you've listened for at least 30 seconds
4. Look for the toast notification: "Auto-liked! (Listened 30+ seconds)"

### Songs Skipping Unexpectedly

**Problem**: Songs end early or skip when paused

**Solution**:
1. The pause button should not skip songs - this was fixed
2. Songs only switch when there are less than 10 seconds remaining
3. If songs are still ending early, check:
   - Your Spotify app is working correctly
   - The song isn't being manually skipped elsewhere

### Brain State Keeps Resetting

**Problem**: Brain state resets to "sad" on every update

**Solution**:
1. This was fixed - brain state now persists across reruns
2. Brain state only updates when you change the sidebar mood selection
3. If it's still resetting, try:
   - Refreshing the browser
   - Starting a new session

### Queued Song Not Playing

**Problem**: Queued song doesn't play when current song ends

**Solution**:
1. Queued songs have priority and should play automatically
2. Check that:
   - A song is actually queued (you should see "Next in Queue" section)
   - The current song has actually finished (less than 10s remaining)
   - Spotify is still active and playing

---

## Technical Details

### Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Python with Bayesian optimization
- **API**: Spotify Web API via `spotipy`
- **Brain Simulation**: Simulated EEG signal processing

### Update Cycle

- The app automatically reruns every **10 seconds** when a session is active
- This ensures:
  - Progress bar updates
  - Auto-like checks run
  - Song end detection works
  - Queue management functions

### Data Flow

1. User selects mood → Brain state simulated → Mood classified
2. Mood → Bayesian optimizer → Song selected
3. Song plays → User feedback (skip/like) → Optimizer learns
4. Next song suggested → Queued → Plays when current ends

---

## Future Enhancements

- [ ] Integration with real EEG devices (Muse, OpenBCI)
- [ ] Real-time mood detection from actual brainwaves
- [ ] Playlist generation based on learned preferences
- [ ] Multi-user support with separate preference profiles
- [ ] Advanced mood detection with multiple emotional states

---

## Support

If you encounter issues not covered in this guide:

1. Check the **Troubleshooting** section above
2. Verify your Spotify Premium account is active
3. Ensure all dependencies are installed correctly
4. Check that your `.env` file is properly configured

---

## License

This project is for educational and personal use. Spotify API usage is subject to Spotify's Terms of Service.

---

**Enjoy your personalized music experience with NeuroDJ! 🎵🧠**
