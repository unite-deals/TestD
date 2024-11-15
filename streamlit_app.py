import streamlit as st
import numpy as np
import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib

hide_github_link_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visiblity: hidden;}
    header {visibility: hidden;}
        .viewerBadge_container__1QSob {
            display: none !important;
        }
    </style>
"""
st.markdown(hide_github_link_style, unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


# Constants
IPD = 6.5
MONITOR_W = 38.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
@st.cache_resource
def load_model():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    }
    depth_model = DepthAnythingV2(**model_configs['vits'])
    checkpoint_path = 'checkpoints/depth_anything_v2_vits.pth'
    depth_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return depth_model.to(DEVICE).eval()

depth_model = load_model()

def process_image(uploaded_image, input_size):
    """Process the uploaded image and predict depth."""
    img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (input_size, input_size))
    depth = depth_model.infer_image(img_resized, input_size)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_colormap = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)
    return img, depth_colormap

# Streamlit App
#st.set_page_config(page_title="Depth Map Prediction", layout="wide")

# UI Layout
st.title("Depth Map Prediction Application")
st.markdown("""
    This app allows you to upload an image and predicts its depth map using advanced AI models.
    - Supports high-quality depth map generation.
    - Outputs a side-by-side comparison of the original image and depth map.
""")

# Sidebar
st.sidebar.header("Configuration")
input_size = st.sidebar.slider("Input Size", 256, 1024, 518, step=32)

# File Upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.subheader("Original Image and Depth Map")
    with st.spinner("Processing..."):
        original, depth_map = process_image(uploaded_file, input_size)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original Image", use_column_width=True)
    with col2:
        st.image(depth_map, caption="Depth Map", use_column_width=True)
    
    # Download Option
    _, col, _ = st.columns([1, 2, 1])
    with col:
        depth_map_file = cv2.imencode('.png', depth_map)[1].tobytes()
        st.download_button("Download Depth Map", depth_map_file, file_name="depth_map.png")

st.info("For any feedback or issues, contact support@example.com")
