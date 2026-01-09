# NeuroDJ

A brain-controlled music recommendation system that uses Bayesian optimization and lyrical analysis to curate personalized playlists based on your emotional state.

## What It Does

NeuroDJ combines **audio features** (valence, energy) with **lyrical analysis** (complexity, bridge shifts, archetypes) to create a sophisticated music recommendation engine. The system:

- **Reads your brain state** (simulated EEG signals) to detect your current mood
- **Applies mood-based filters** to song selection:
  - **Focus mode**: Filters out complex lyrics that might distract
  - **Boredom rescue**: Prioritizes songs with dramatic bridge shifts
  - **Sadness mode**: Excludes overly upbeat "Glitter Gel Pen" songs
- **Learns your preferences** through Bayesian optimization as you listen
- **Auto-queues songs** based on your feedback and mood changes
- **Integrates with Spotify** for seamless playback control

## Key Features

- **Dual-Feature Matching**: Combines audio features (valence/energy) with lyrical characteristics
- **Mood-Aware Filtering**: Applies lyrical filters based on detected brain state
- **Adaptive Learning**: Uses Bayesian optimization to improve recommendations over time
- **Real-time Playback**: Direct Spotify integration for instant song control
- **Auto-Like System**: Automatically registers positive feedback after 30+ seconds of listening

## How It Works

1. **Brain State Detection**: Simulates EEG signals and classifies your emotional state (sad, happy, anger, focus)
2. **Song Selection**: The optimizer suggests audio features (valence, energy) while lyrical filters constrain the search space
3. **Feedback Loop**: Your reactions (skip/like) teach the AI your preferences
4. **Continuous Adaptation**: The system gets smarter with each interaction

## Technology Stack

- **Backend**: Python with Bayesian optimization (`bayesian-optimization`)
- **Frontend**: Streamlit web interface
- **Music API**: Spotify Web API (`spotipy`)
- **Data**: Merged dataset combining audio features and lyrical analysis (`neurodj_data.csv`)
