import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import io
import sys
import os
import tempfile
import subprocess

# Version check
if sys.version_info >= (3, 12):
    st.error("This app requires Python 3.11. Please contact support.")
    st.stop()

# App Configuration
st.set_page_config(
    page_title="GlowMatch Pro",
    page_icon="💄",
    layout="centered"
)

# Constants
MAX_FILE_SIZE_MB = 5
REPO_URL = "https://github.com/yourusername/yourrepo.git"  # Update this

# Try to import MediaPipe with fallback
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    FACE_MESH = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
except ImportError:
    st.error("MediaPipe failed to load. Using fallback detection.")
    FACE_MESH = None

# Asset loading functions
@st.cache_data
def load_asset(branch, filename):
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            subprocess.run([
                "git", "clone",
                "--branch", branch,
                "--depth", "1",
                "--single-branch",
                REPO_URL,
                tmp_dir
            ], check=True, capture_output=True)
            
            asset_path = os.path.join(tmp_dir, filename)
            if os.path.exists(asset_path):
                img = Image.open(asset_path).convert("RGBA")
                alpha = img.split()[-1].filter(ImageFilter.GaussianBlur(radius=1))
                img.putalpha(alpha)
                return img
        except Exception as e:
            st.error(f"Error loading asset: {str(e)}")
    return None

# Main App
def main():
    st.title("✨ GlowMatch Pro")
    
    uploaded_file = st.file_uploader("Upload a selfie", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            img = Image.open(uploaded_file).convert("RGBA")
            
            if FACE_MESH:
                results = FACE_MESH.process(np.array(img.convert("RGB")))
                if not results.multi_face_landmarks:
                    st.warning("No face detected. Using default positions.")
                    landmarks = None
                else:
                    landmarks = results.multi_face_landmarks[0]
            else:
                landmarks = None
                st.warning("Using fallback mode - facial features may not align perfectly")

            # Rest of your application logic here
            # [Previous code for applying makeup overlays]

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
