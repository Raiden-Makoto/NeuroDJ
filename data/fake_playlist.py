DUMMY_PLAYLIST = [
    {"title": "Blank Space", "artist": "Taylor Swift",      "valence": 0.9, "energy": 0.8}, # Happy
    {"title": "Red",               "artist": "Taylor Swift",     "valence": 0.95, "energy": 0.8}, # Happy
    {"title": "Back to December",    "artist": "Taylor Swift", "valence": 0.3, "energy": 0.2}, # Sad
    {"title": "All Too Well",    "artist": "Taylor Swift",        "valence": 0.2, "energy": 0.3}, # Sad
    {"title": "Say Don't Go",    "artist": "Taylor Swift",        "valence": 0.3, "energy": 0.3}, # Sad
    {"title": "Mean",   "artist": "Taylor Swift",    "valence": 0.2, "energy": 0.9}, # Angry
    {"title": "You Need to Calm Down",     "artist": "Taylor Swift", "valence": 0.15, "energy": 0.95},# Angry
    {"title": "State of Grace",          "artist": "Taylor Swift","valence": 0.8, "energy": 0.1}, # Calm
    {"title": "Nocturne in E-flat major, Op. 9, No. 2", "artist": "Frédéric Chopin", "valence": 0.7, "energy": 0.3}, # Calm
    {"title": "Love Story",      "artist": "Taylor Swift",  "valence": 0.4, "energy": 0.9}, # Mixed
    {"title": "Welcome to New York",   "artist": "Taylor Swift",        "valence": 0.5, "energy": 0.6}, # Neutral
]

USER_TASTE_PROFILE = {
    "sad":   {"target_v": 0.25, "target_e": 0.25}, 
    "happy": {"target_v": 0.90, "target_e": 0.80}, 
    "anger": {"target_v": 0.20, "target_e": 0.90}, 
    "neutral": {"target_v": 0.50, "target_e": 0.50}
}