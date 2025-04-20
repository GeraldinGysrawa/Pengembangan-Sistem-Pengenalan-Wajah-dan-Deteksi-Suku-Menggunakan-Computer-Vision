import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import time
from face_similarity import FaceSimilarity

# Set page config
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Capture Faces", "Preprocess Dataset", "Face Similarity"])

class FacePreprocessor:
    def __init__(self, input_dataset_path, output_dataset_path, metadata_path):
        self.input_dataset_path = Path(input_dataset_path)
        self.output_dataset_path = Path(output_dataset_path)
        self.metadata_path = Path(metadata_path)
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dataset_path, exist_ok=True)
        
        # Load metadata
        self.metadata = pd.read_csv(self.metadata_path) if self.metadata_path.exists() else None

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            return faces[0]  # Return the first face detected
        return None

    def normalize_face(self, image, face_bbox, target_size=(224, 224)):
        x, y, w, h = face_bbox
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        return face

    def augment_image(self, image, do_flip=True, do_rotate=True, 
                     do_brightness=True, do_noise=True):
        augmented_images = [image]  # Original image is always included
        
        if do_flip:
            # Horizontal flip
            flipped = cv2.flip(image, 1)
            augmented_images.append(flipped)
        
        if do_rotate:
            # Random rotation (±15 degrees)
            angle = np.random.uniform(-15, 15)
            height, width = image.shape[:2]
            center = (width/2, height/2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
            augmented_images.append(rotated)
        
        if do_brightness:
            # Brightness adjustment
            brightness_factor = np.random.uniform(0.8, 1.2)
            brightened = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
            augmented_images.append(brightened)
        
        if do_noise:
            # Add Gaussian noise
            noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
            noisy = cv2.add(image, noise)
            augmented_images.append(noisy)
        
        return augmented_images

    def process_dataset(self, target_size=(224, 224), do_flip=True, 
                       do_rotate=True, do_brightness=True, do_noise=True):
        stats = {
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "people": set(),
            "ethnicities": set(),
            "augmented_images": 0
        }
        
        # Process each person's directory
        for person_dir in os.listdir(self.input_dataset_path):
            person_path = os.path.join(self.input_dataset_path, person_dir)
            if not os.path.isdir(person_path):
                continue
                
            stats["people"].add(person_dir)
            
            # Process each ethnicity subdirectory
            for ethnicity_dir in os.listdir(person_path):
                ethnicity_path = os.path.join(person_path, ethnicity_dir)
                if not os.path.isdir(ethnicity_path):
                    continue
                    
                stats["ethnicities"].add(ethnicity_dir)
                
                # Create corresponding output directories
                output_person_path = os.path.join(self.output_dataset_path, person_dir)
                output_ethnicity_path = os.path.join(output_person_path, ethnicity_dir)
                os.makedirs(output_ethnicity_path, exist_ok=True)
                
                # Process each image
                for img_file in os.listdir(ethnicity_path):
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    stats["total_images"] += 1
                    img_path = os.path.join(ethnicity_path, img_file)
                    
                    try:
                        # Read and process image
                        image = cv2.imread(img_path)
                        if image is None:
                            raise ValueError("Could not read image")
                            
                        # Detect face
                        face_bbox = self.detect_face(image)
                        if face_bbox is None:
                            raise ValueError("No face detected")
                            
                        # Normalize face
                        face = self.normalize_face(image, face_bbox, target_size)
                        
                        # Save original processed face
                        output_path = os.path.join(output_ethnicity_path, img_file)
                        cv2.imwrite(output_path, face)
                        stats["processed_images"] += 1
                        
                        # Apply augmentations
                        augmented_faces = self.augment_image(
                            face, do_flip, do_rotate, do_brightness, do_noise
                        )
                        
                        # Save augmented faces
                        for i, aug_face in enumerate(augmented_faces[1:], 1):  # Skip original
                            aug_filename = f"{os.path.splitext(img_file)[0]}_aug_{i}{os.path.splitext(img_file)[1]}"
                            aug_path = os.path.join(output_ethnicity_path, aug_filename)
                            cv2.imwrite(aug_path, aug_face)
                            stats["augmented_images"] += 1
                            
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
                        stats["failed_images"] += 1
        
        return stats

    def visualize_processing(self, image_path, do_flip=True, do_rotate=True, 
                           do_brightness=True, do_noise=True):
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Could not read image")
            
        # Create figure with subplots (2 rows, 4 columns)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle("Face Processing Steps", fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Face detection
        face_bbox = self.detect_face(image)
        if face_bbox is not None:
            x, y, w, h = face_bbox
            detection_img = image.copy()
            cv2.rectangle(detection_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            axes[0, 1].imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title("Face Detection")
            axes[0, 1].axis('off')
            
            # Normalized face
            face = self.normalize_face(image, face_bbox)
            axes[0, 2].imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            axes[0, 2].set_title("Normalized Face")
            axes[0, 2].axis('off')
            
            # Get augmented faces
            augmented_faces = self.augment_image(
                face, do_flip, do_rotate, do_brightness, do_noise
            )
            
            # Plot selected augmentations
            plot_idx = 0
            aug_idx = 1  # Start from index 1 since 0 is the original image
            
            # Create a list of augmentation options to process
            aug_options = [
                (do_flip, "Horizontal Flip"),
                (do_rotate, "Random Rotation"),
                (do_brightness, "Brightness Adjustment"),
                (do_noise, "Gaussian Noise")
            ]
            
            # Process each augmentation option
            for is_enabled, title in aug_options:
                if is_enabled and aug_idx < len(augmented_faces):
                    row = 1
                    col = plot_idx
                    axes[row, col].imshow(cv2.cvtColor(augmented_faces[aug_idx], cv2.COLOR_BGR2RGB))
                    axes[row, col].set_title(title)
                    axes[row, col].axis('off')
                    plot_idx += 1
                    aug_idx += 1
            
            # Hide unused subplots
            for i in range(plot_idx, 4):
                axes[1, i].axis('off')
        else:
            for i in range(8):  # 2 rows x 4 columns
                row = i // 4
                col = i % 4
                ax = axes[row, col]
                ax.text(0.5, 0.5, "No face detected", 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes)
                ax.axis('off')
        
        plt.tight_layout()
        return fig

if page == "Capture Faces":
    # Title and description
    st.title("👤 Add New Faces to Dataset")
    st.markdown("""
    This page allows you to capture and add new face images to the dataset.
    Make sure to follow the dataset structure and requirements.
    """)

    # Input nama orang baru
    new_person = st.text_input("Enter new person's name:")

    # Input suku
    ethnicity = st.selectbox(
        "Select ethnicity:",
        ["Jawa", "Sunda", "Batak", "Minang", "Other"]
    )

    # Tombol untuk memulai proses penambahan wajah
    capture = st.button("Start Capturing")

    if capture:
        if not new_person:
            st.warning("Please enter a name for the new person.")
        else:
            # Create directory structure
            save_path = os.path.join('dataset', new_person, ethnicity)
            os.makedirs(save_path, exist_ok=True)
            
            if not os.path.exists('dataset'):
                os.makedirs('dataset')
                st.info("Created 'dataset' folder.")
            
            st.success(f"Created folder for {new_person} ({ethnicity})")
            
            # Mulai menangkap gambar dari webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam. Please check your camera connection.")
            else:
                num_images = 0
                max_images = 20  # Ambil 20 gambar wajah

                frame_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    while num_images < max_images:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Error: Cannot read frame from webcam.")
                            break

                        # Deteksi wajah dalam frame
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(
                            gray, 
                            scaleFactor=1.1, 
                            minNeighbors=5, 
                            minSize=(30, 30)
                        )

                        if len(faces) > 0:
                            for (x, y, w, h) in faces:
                                face = frame[y:y+h, x:x+w]
                                img_name = os.path.join(save_path, f"img_{num_images}.jpg")
                                cv2.imwrite(img_name, face)
                                num_images += 1

                                # Menggambar kotak di sekitar wajah yang terdeteksi
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                                # Tampilkan hasil deteksi
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(frame_rgb, channels="RGB", caption=f"Image {num_images}/{max_images}")

                                # Update progress bar
                                progress = num_images / max_images
                                progress_bar.progress(progress)
                                status_text.text(f"Saving image {num_images} of {max_images}...")

                                # Hentikan setelah menyimpan satu wajah per frame
                                break
                        else:
                            # Tampilkan frame tanpa deteksi
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", caption="No face detected.")

                        time.sleep(0.1)  # Tambahkan delay untuk menghindari penggunaan CPU yang berlebihan

                    st.success(f"Successfully added {num_images} images to {new_person}'s dataset.")
                finally:
                    cap.release()
                    frame_placeholder.empty()
                    progress_bar.empty()
                    status_text.empty()

elif page == "Face Similarity":
    st.title("Face Similarity")
    
    # Inisialisasi FaceSimilarity
    face_similarity = FaceSimilarity()
    
    # Upload dua gambar untuk dibandingkan
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar 1")
        image1 = st.file_uploader("Upload gambar pertama", type=['jpg', 'jpeg', 'png'])
        if image1 is not None:
            st.image(image1, use_column_width=True)
            
    with col2:
        st.subheader("Gambar 2")
        image2 = st.file_uploader("Upload gambar kedua", type=['jpg', 'jpeg', 'png'])
        if image2 is not None:
            st.image(image2, use_column_width=True)
    
    # Tampilkan threshold yang digunakan
    st.info(f"Threshold yang digunakan: {face_similarity.similarity_threshold:.2f}")
    st.caption("Jika similarity score ≥ threshold, wajah dianggap SAMA")
    st.caption("Jika similarity score < threshold, wajah dianggap BERBEDA")
    
    # Tombol untuk membandingkan wajah
    if st.button("Bandingkan Wajah") and image1 is not None and image2 is not None:
        try:
            # Simpan gambar sementara
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            image1_path = temp_dir / "temp1.jpg"
            image2_path = temp_dir / "temp2.jpg"
            
            with open(image1_path, "wb") as f:
                f.write(image1.getvalue())
            with open(image2_path, "wb") as f:
                f.write(image2.getvalue())
            
            # Proses gambar
            embedding1 = face_similarity.process_image(image1_path)
            embedding2 = face_similarity.process_image(image2_path)
            
            if embedding1 is None or embedding2 is None:
                st.error("Tidak dapat mendeteksi wajah pada salah satu atau kedua gambar")
            else:
                # Hitung similarity score
                similarity_score = face_similarity.compare_faces(embedding1, embedding2)
                
                # Visualisasi hasil
                result_image = face_similarity.visualize_comparison(
                    str(image1_path), 
                    str(image2_path), 
                    similarity_score
                )
                
                # Tampilkan hasil
                st.image(result_image, use_column_width=True)
                
                # Tampilkan similarity score dan perbandingan dengan threshold
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Similarity Score", f"{similarity_score:.2f}")
                with col2:
                    st.metric("Threshold", f"{face_similarity.similarity_threshold:.2f}")
                
                # Tentukan apakah wajah sama
                if similarity_score >= face_similarity.similarity_threshold:
                    st.success(f"Wajah SAMA (Score {similarity_score:.2f} ≥ Threshold {face_similarity.similarity_threshold:.2f})")
                else:
                    st.error(f"Wajah BERBEDA (Score {similarity_score:.2f} < Threshold {face_similarity.similarity_threshold:.2f})")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
        finally:
            # Hapus file temporary
            if os.path.exists(image1_path):
                os.remove(image1_path)
            if os.path.exists(image2_path):
                os.remove(image2_path)

else:  # Preprocess Dataset page
    # Title and description
    st.title("🔄 Dataset Preprocessing Tool")
    st.markdown("""
    This tool helps you preprocess your dataset for face recognition and ethnic detection.
    It performs face detection, normalization, and augmentation on your images.
    """)

    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        input_dataset = st.text_input(
            "Input Dataset Path",
            value="dataset",
            help="Path to the input dataset directory"
        )
        
        metadata_path = st.text_input(
            "Metadata CSV Path",
            value="dataset/metadata.csv",
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
                stats = preprocessor.process_dataset(
                    target_size=(target_size, target_size),
                    do_flip=horizontal_flip,
                    do_rotate=random_rotation,
                    do_brightness=brightness_adjust,
                    do_noise=gaussian_noise
                )
                
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
                
                # Visualize processing with selected augmentation options
                fig = preprocessor.visualize_processing(
                    selected_image,
                    do_flip=horizontal_flip,
                    do_rotate=random_rotation,
                    do_brightness=brightness_adjust,
                    do_noise=gaussian_noise
                )
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
