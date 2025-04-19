import streamlit as st
from face_capture import show_face_capture_page
from dataset_preprocessing import show_dataset_preprocessing_page
from ethnicity_detection import show_ethnicity_detection_page

# Set page config
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Capture Faces", "Preprocess Dataset", "Ethnicity Detection"])

# Show selected page
if page == "Capture Faces":
    show_face_capture_page()
elif page == "Preprocess Dataset":
    show_dataset_preprocessing_page()
elif page == "Ethnicity Detection":
    show_ethnicity_detection_page()
