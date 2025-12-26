import streamlit as st #type: ignore
import pandas as pd #type: ignore

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Neuro-DJ", page_icon="🧠", layout="centered")

# Custom CSS to force "Spotify Dark Mode" look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #121212;
        color: white;
    }
    /* Buttons (Green Spotify Style) */
    .stButton>button {
        background-color: #1DB954;
        color: #FFFFFF !important;
        border-radius: 50px;
        border: none;
        padding: 12px 30px;
        font-weight: bold;
        font-size: 16px;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background-color: #1ed760;
        transform: scale(1.05);
        color: #FFFFFF !important;
    }
    /* Headings */
    h1, h2, h3 { font-family: 'Circular', sans-serif; color: white; }
    p { color: #b3b3b3; }
    
    /* Cards */
    .css-1r6slb0 {
        background-color: #181818;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- MOCK DATA (Static State) ---
if 'current_track' not in st.session_state:
    st.session_state.current_track = {
        "title": "Anti-Hero",
        "artist": "Taylor Swift",
        "album": "Midnights",
        "cover": "https://upload.wikimedia.org/wikipedia/en/9/9f/Midnights_-_Taylor_Swift.png",
        "time": "0:45 / 3:20"
    }

if 'history' not in st.session_state:
    st.session_state.history = [
        {"Title": "Lavender Haze", "Artist": "Taylor Swift", "Mood": "FOCUS", "Reaction": "Liked"},
        {"Title": "Maroon", "Artist": "Taylor Swift", "Mood": "SAD", "Reaction": "Skipped"},
        {"Title": "Snow on the Beach", "Artist": "Taylor Swift", "Mood": "SAD", "Reaction": "Liked"},
    ]

# --- DUMMY FUNCTIONS (Do nothing for now) ---
def mock_play_pause():
    print("Clicked Play/Pause")

def mock_skip():
    # In the real app, this would call the AI
    print("Clicked Skip")
    # Just for demo fun, let's swap the title
    if st.session_state.current_track['title'] == "Anti-Hero":
        st.session_state.current_track['title'] = "Midnight Rain"
    else:
        st.session_state.current_track['title'] = "Anti-Hero"

def mock_like():
    print("Clicked Like")

# --- UI LAYOUT ---

# 1. HEADER
st.title("🧠 Neuro-DJ")
st.markdown("**Connected Device:** *MacBook Pro (Spotify Desktop)*")
st.divider()

# 2. MAIN PLAYER (The "Now Playing" Card)
col_art, col_info = st.columns([1, 2])

with col_art:
    # Album Art
    st.image(st.session_state.current_track['cover'], width=200)

with col_info:
    # Song Details
    st.markdown(f"## {st.session_state.current_track['title']}")
    st.markdown(f"### {st.session_state.current_track['artist']}")
    st.markdown(f"*{st.session_state.current_track['album']}*")
    
    # Fake Progress Bar
    st.slider("Progress", 0, 100, 30, label_visibility="collapsed")
    st.caption(f"{st.session_state.current_track['time']}")

# 3. CONTROLS
st.write("") # Spacer
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.button("SKIP", on_click=mock_skip, width='stretch')
with c2:
    st.button("PAUSE", on_click=mock_play_pause, width='stretch')
with c3:
    st.button("LIKE", on_click=mock_like, width='stretch')

# 4. HISTORY LIST
st.divider()
st.subheader("🎵 Session History")

# Create a clean table for history
df_hist = pd.DataFrame(st.session_state.history)
st.dataframe(
    df_hist, 
    width='stretch', 
    hide_index=True,
    column_config={
        "Reaction": st.column_config.TextColumn(
            "Reaction",
            help="Did you keep or skip?",
            width="small"
        )
    }
)