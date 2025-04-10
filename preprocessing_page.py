import streamlit as st
import os
import matplotlib.pyplot as plt
from preprocessing import FacePreprocessor

def show_preprocessing_page():
    """
    Display the face preprocessing page in the Streamlit app.
    """
    # Title and description
    st.title("🔄 Face Preprocessing")
    st.markdown("""
    This tool performs face detection, normalization, and augmentation on your dataset.
    It helps prepare your images for face recognition and ethnic detection.
    """)
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        input_dataset = st.text_input(
            "Input Dataset Path",
            value="dataset_tubes",
            help="Path to the input dataset directory"
        )
        
        metadata_path = st.text_input(
            "Metadata CSV Path",
            value="dataset_tubes/metadata.csv",
            help="Path to the metadata CSV file"
        )
    
    with col2:
        output_dataset = st.text_input(
            "Output Dataset Path",
            value="processed_dataset",
            help="Path where processed images will be saved"
        )
        
        target_size = st.slider(
            "Target Face Size",
            min_value=64,
            max_value=512,
            value=224,
            step=32,
            help="Size to resize detected faces to"
        )
    
    # Augmentation options
    st.subheader("Augmentation Options")
    col1, col2 = st.columns(2)
    
    with col1:
        horizontal_flip = st.checkbox("Horizontal Flip", value=True)
        random_rotation = st.checkbox("Random Rotation", value=True)
    
    with col2:
        brightness_adjust = st.checkbox("Brightness Adjustment", value=True)
        gaussian_noise = st.checkbox("Gaussian Noise", value=True)
    
    # Process button
    if st.button("Process Dataset"):
        if not os.path.exists(input_dataset):
            st.error(f"Input dataset directory '{input_dataset}' does not exist!")
        else:
            with st.spinner("Processing dataset..."):
                # Initialize preprocessor
                preprocessor = FacePreprocessor(
                    input_dataset_path=input_dataset,
                    output_dataset_path=output_dataset,
                    metadata_path=metadata_path
                )
                
                # Process dataset
                stats = preprocessor.process_dataset()
                
                # Display results
                st.success("Dataset processing completed!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Images", stats["total_images"])
                with col2:
                    st.metric("Processed Images", stats["processed_images"])
                with col3:
                    st.metric("Failed Images", stats["failed_images"])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("People", len(stats["people"]))
                with col2:
                    st.metric("Ethnicities", len(stats["ethnicities"]))
                with col3:
                    st.metric("Augmented Images", stats["augmented_images"])
    
    # Visualization section
    st.header("Visualization")
    
    # Get list of images for visualization
    if os.path.exists(input_dataset):
        all_images = []
        for root, _, files in os.walk(input_dataset):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(root, file))
        
        if all_images:
            selected_image = st.selectbox(
                "Select an image to visualize processing steps:",
                all_images
            )
            
            if selected_image:
                # Initialize preprocessor for visualization
                preprocessor = FacePreprocessor(
                    input_dataset_path=input_dataset,
                    output_dataset_path=output_dataset,
                    metadata_path=metadata_path
                )
                
                # Visualize processing
                fig = preprocessor.visualize_processing(selected_image)
                st.pyplot(fig)
        else:
            st.warning("No images found in the input dataset directory.")
    else:
        st.warning("Input dataset directory does not exist.")
    
    # Instructions
    st.header("Instructions")
    st.markdown("""
    1. Enter the paths for your input dataset, metadata file, and output directory
    2. Adjust the target face size and augmentation options as needed
    3. Click 'Process Dataset' to start the preprocessing
    4. Use the visualization section to see how the processing affects individual images
    """) 