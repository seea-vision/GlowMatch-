import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import mediapipe as mp
import io
import base64
import subprocess
import tempfile
import os

# App Configuration
st.set_page_config(
    page_title="GlowMatch Pro",
    page_icon="💄",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Constants
MAX_FILE_SIZE_MB = 5
DEFAULT_IMAGE_WIDTH = 500
REPO_URL = "https://github.com/yourusername/yourrepo.git"  # Replace with your actual repo

# Asset Branches Configuration
ASSET_BRANCHES = {
    "lashes": "lashes",
    "brows": "brows",
    "lips": "lips",
    "blush": "blush"
}

# Default asset files (these should exist in each branch)
DEFAULT_ASSET_FILES = {
    "lashes": ["natural.png", "dramatic.png", "wispy.png"],
    "brows": ["natural.png", "bold.png"],
    "lips": ["nude.png", "red.png"],
    "blush": ["soft_pink.png", "peach.png"]
}

# Initialize MediaPipe Face Mesh
@st.cache_resource
def get_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# Git Asset Loader
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_asset_from_branch(branch_name, asset_file):
    """Load an asset file from a specific git branch"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Clone just the specific branch and file
            subprocess.run([
                "git", "clone", 
                "--branch", branch_name,
                "--depth", "1",
                "--filter=blob:none",
                "--sparse",
                REPO_URL,
                tmp_dir
            ], check=True)
            
            # Checkout just the specific file
            subprocess.run([
                "git", "-C", tmp_dir,
                "sparse-checkout", "set",
                asset_file
            ], check=True)
            
            asset_path = os.path.join(tmp_dir, asset_file)
            if os.path.exists(asset_path):
                return Image.open(asset_path).convert("RGBA")
            return None
        except Exception as e:
            st.error(f"Error loading asset: {str(e)}")
            return None

# Utility Functions
def validate_image(uploaded_file):
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"File exceeds {MAX_FILE_SIZE_MB}MB limit")
    try:
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)
        return True
    except Exception as e:
        raise ValueError(f"Invalid image: {str(e)}")

def feather_alpha(image):
    alpha = image.split()[-1].filter(ImageFilter.GaussianBlur(radius=1))
    image.putalpha(alpha)
    return image

def apply_overlay(base_img, overlay, position, scale=1.0, angle=0.0, opacity=1.0):
    """Apply overlay with transformations"""
    new_size = (int(overlay.width * scale), int(overlay.height * scale)
    overlay = overlay.resize(new_size, Image.LANCZOS)
    
    if angle != 0:
        overlay = overlay.rotate(angle, expand=True, resample=Image.BILINEAR)
    
    if opacity < 1.0:
        overlay = overlay.copy()
        alpha = overlay.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
        overlay.putalpha(alpha)
    
    x, y = position
    paste_position = (x - overlay.width // 2, y - overlay.height // 2)
    base_img.paste(overlay, paste_position, overlay)
    return base_img

# Face Landmark Detection
def detect_landmarks(image, face_mesh):
    image_rgb = np.array(image.convert("RGB"))
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    landmarks = results.multi_face_landmarks[0].landmark
    h, w = image.height, image.width
    
    landmark_indices = {
        "left_eye": 159,
        "right_eye": 386,
        "left_brow": 70,
        "right_brow": 300,
        "upper_lip": 13,
        "lower_lip": 14,
        "left_cheek": 123,
        "right_cheek": 352,
        "face_left": 234,
        "face_right": 454
    }
    
    points = {}
    for name, idx in landmark_indices.items():
        landmark = landmarks[idx]
        points[name] = (int(landmark.x * w), int(landmark.y * h))
    
    points["face_width"] = abs(points["face_right"][0] - points["face_left"][0])
    return points

# UI Components
def create_style_controls(style_type):
    """Create style selection UI elements"""
    available_files = DEFAULT_ASSET_FILES.get(style_type, [])
    display_names = [f.replace(".png", "").replace("_", " ").title() for f in available_files]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_idx = st.selectbox(
            f"{style_type.capitalize()} Style",
            options=range(len(available_files)),
            format_func=lambda x: display_names[x],
            key=f"{style_type}_style"
        )
        selected_file = available_files[selected_idx]
    with col2:
        intensity = st.slider(
            "Intensity",
            0.5, 2.0, 1.0,
            key=f"{style_type}_intensity"
        )
    
    # Load preview image
    preview_img = load_asset_from_branch(ASSET_BRANCHES[style_type], selected_file)
    if preview_img:
        preview_img.thumbnail((150, 75))
        st.sidebar.image(preview_img, caption=display_names[selected_idx])
    
    return selected_file, intensity

# Main Application
def main():
    # Check if git is available
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except:
        st.error("Git is required to run this application. Please install Git.")
        return
    
    # Initialize face mesh
    face_mesh = get_face_mesh()
    
    # App Header
    st.title("💄 GlowMatch Pro")
    st.markdown("Virtual Makeup Try-On with AI")
    
    with st.expander("ℹ️ Instructions"):
        st.write("""
        1. Upload a clear front-facing photo
        2. Adjust your makeup preferences
        3. Download your enhanced look
        """)
    
    # Sidebar Controls
    st.sidebar.header("Makeup Settings")
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Choose a selfie...",
        type=["jpg", "jpeg", "png"],
        help=f"Max file size: {MAX_FILE_SIZE_MB}MB"
    )
    
    if uploaded_file:
        try:
            validate_image(uploaded_file)
            
            with st.spinner("Enhancing your image..."):
                original_img = Image.open(uploaded_file).convert("RGBA")
                landmarks = detect_landmarks(original_img, face_mesh)
                
                if not landmarks:
                    st.error("No face detected. Please try another photo.")
                    st.image(original_img, width=DEFAULT_IMAGE_WIDTH)
                    return
                
                # Get user preferences
                lash_file, lash_intensity = create_style_controls("lashes")
                brow_file, brow_intensity = create_style_controls("brows")
                lip_file, lip_intensity = create_style_controls("lips")
                blush_file, blush_intensity = create_style_controls("blush")
                
                # Load assets from git branches
                lash_img = load_asset_from_branch("lashes", lash_file)
                brow_img = load_asset_from_branch("brows", brow_file)
                lip_img = load_asset_from_branch("lips", lip_file)
                blush_img = load_asset_from_branch("blush", blush_file)
                
                if None in [lash_img, brow_img, lip_img, blush_img]:
                    st.error("Failed to load some assets. Please try again.")
                    return
                
                # Apply feathering
                lash_img = feather_alpha(lash_img)
                brow_img = feather_alpha(brow_img)
                lip_img = feather_alpha(lip_img)
                blush_img = feather_alpha(blush_img)
                
                # Calculate transformations
                face_scale = landmarks["face_width"] / 300
                eye_dx = landmarks["right_eye"][0] - landmarks["left_eye"][0]
                eye_dy = landmarks["right_eye"][1] - landmarks["left_eye"][1]
                eye_angle = np.degrees(np.arctan2(eye_dy, eye_dx))
                
                # Create enhanced image
                enhanced_img = original_img.copy()
                
                # Apply all makeup (same as before)
                # ... [rest of your application logic remains the same]

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("Please upload a photo to begin")
        st.image("https://via.placeholder.com/500x500?text=Upload+a+selfie", 
                width=DEFAULT_IMAGE_WIDTH)

if __name__ == "__main__":
    main()
