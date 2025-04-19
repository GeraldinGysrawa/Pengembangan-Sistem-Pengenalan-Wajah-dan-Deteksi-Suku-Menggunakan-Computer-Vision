import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st
from mtcnn import MTCNN

class FacePreprocessor:
    def __init__(self, input_dataset_path, output_dataset_path, metadata_path):
        self.input_dataset_path = Path(input_dataset_path)
        self.output_dataset_path = Path(output_dataset_path)
        self.metadata_path = Path(metadata_path)
        
        # Initialize face detector using MTCNN
        self.detector = MTCNN()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dataset_path, exist_ok=True)
        
        # Load metadata
        self.metadata = pd.read_csv(self.metadata_path) if self.metadata_path.exists() else None

    def detect_face(self, image):
        """Detect face using MTCNN with enhanced confidence threshold"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Detect faces with MTCNN
            faces = self.detector.detect_faces(image)
            
            if not faces:
                st.warning("No faces detected in image")
                return None
            
            # Get the face with highest confidence
            face = max(faces, key=lambda x: x['confidence'])
            
            # Check confidence threshold
            if face['confidence'] < 0.95:  # Increased threshold for better quality
                st.warning(f"Face detection confidence too low: {face['confidence']:.2f}")
                return None
            
            # Extract face box and landmarks
            x, y, w, h = face['box']
            left_eye = face['keypoints']['left_eye']
            right_eye = face['keypoints']['right_eye']
            
            return {
                'box': (x, y, w, h),
                'confidence': face['confidence'],
                'landmarks': {
                    'left_eye': left_eye,
                    'right_eye': right_eye
                }
            }
            
        except Exception as e:
            st.error(f"Error in face detection: {str(e)}")
            return None

    def align_face(self, image, face_data):
        """Align face using eye landmarks"""
        try:
            # Get eye coordinates
            left_eye = face_data['landmarks']['left_eye']
            right_eye = face_data['landmarks']['right_eye']
            
            # Calculate angle for alignment
            eye_angle = np.degrees(np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            ))
            
            # Get face box
            x, y, w, h = face_data['box']
            
            # Calculate center of face
            center = (x + w//2, y + h//2)
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, eye_angle, 1.0)
            
            # Perform rotation
            aligned_face = cv2.warpAffine(
                image,
                rotation_matrix,
                (image.shape[1], image.shape[0]),
                flags=cv2.INTER_CUBIC
            )
            
            # Re-detect face in aligned image
            aligned_face_data = self.detect_face(aligned_face)
            if aligned_face_data is None:
                return None
                
            # Extract aligned face
            x, y, w, h = aligned_face_data['box']
            face = aligned_face[y:y+h, x:x+w]
            
            return face
            
        except Exception as e:
            st.error(f"Error in face alignment: {str(e)}")
            return None

    def normalize_face(self, image, face_data, target_size=(224, 224)):
        """Normalize face with enhanced preprocessing"""
        try:
            # First align the face
            aligned_face = self.align_face(image, face_data)
            if aligned_face is None:
                return None

            # Resize to target size
            face = cv2.resize(aligned_face, target_size, interpolation=cv2.INTER_CUBIC)
            
            # Convert to RGB if needed
            if len(face.shape) == 2:
                face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            
            # Apply histogram equalization to each channel
            face_yuv = cv2.cvtColor(face, cv2.COLOR_RGB2YUV)
            face_yuv[:,:,0] = cv2.equalizeHist(face_yuv[:,:,0])
            face = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2RGB)
            
            return face
        except Exception as e:
            st.error(f"Error in face normalization: {str(e)}")
            return None

    def augment_image(self, image, do_flip=True, do_rotate=True, 
                     do_brightness=True, do_noise=True):
        """Apply data augmentation with controlled number of augmentations"""
        augmented_images = [image]  # Original image is always included
        
        if do_flip:
            # Horizontal flip
            flipped = cv2.flip(image, 1)
            augmented_images.append(flipped)
        
        if do_rotate:
            # Single rotation (±15 degrees)
            angle = np.random.choice([-15, 15])  # Choose one angle
            height, width = image.shape[:2]
            center = (width/2, height/2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
            augmented_images.append(rotated)
        
        if do_brightness:
            # Single brightness adjustment
            alpha = np.random.choice([0.8, 1.2])  # Choose one brightness level
            brightened = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            augmented_images.append(brightened)
        
        if do_noise:
            # Single noise addition (either Gaussian or salt & pepper)
            if np.random.choice([True, False]):
                # Gaussian noise
                sigma = np.random.choice([5, 15])
                noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
                noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
                augmented_images.append(noisy)
            else:
                # Salt and pepper noise
                prob = 0.03
                noisy_sp = image.copy()
                black = np.random.random(image.shape[:2]) < prob/2
                white = np.random.random(image.shape[:2]) < prob/2
                noisy_sp[black] = 0
                noisy_sp[white] = 255
                augmented_images.append(noisy_sp)
        
        return augmented_images

    def process_dataset(self, target_size=(224, 224), do_flip=True, 
                       do_rotate=True, do_brightness=True, do_noise=True):
        """Process and augment the entire dataset"""
        stats = {
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "people": set(),
            "ethnicities": set(),
            "augmented_images": 0
        }
        
        # Create metadata list
        metadata_rows = []
        
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
                        face_data = self.detect_face(image)
                        if face_data is None:
                            raise ValueError("No face detected")
                            
                        # Normalize face
                        face = self.normalize_face(image, face_data, target_size)
                        if face is None:
                            raise ValueError("Face normalization failed")
                        
                        # Save original processed face
                        output_path = os.path.join(output_ethnicity_path, img_file)
                        cv2.imwrite(output_path, face)
                        stats["processed_images"] += 1
                        
                        # Add to metadata
                        metadata_rows.append({
                            'nama': person_dir,
                            'suku': ethnicity_dir,
                            'path_gambar': os.path.join(person_dir, ethnicity_dir, img_file)
                        })
                        
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
                            
                            # Add augmented image to metadata
                            metadata_rows.append({
                                'nama': person_dir,
                                'suku': ethnicity_dir,
                                'path_gambar': os.path.join(person_dir, ethnicity_dir, aug_filename)
                            })
                            
                    except Exception as e:
                        st.error(f"Error processing {img_path}: {str(e)}")
                        stats["failed_images"] += 1
        
        # Save metadata
        if metadata_rows:
            metadata_df = pd.DataFrame(metadata_rows)
            metadata_path = os.path.join(self.output_dataset_path, 'metadata.csv')
            metadata_df.to_csv(metadata_path, index=False)
            st.success(f"Metadata saved to {metadata_path}")
        
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
        face_data = self.detect_face(image)
        if face_data is not None:
            x, y, w, h = face_data['box']
            detection_img = image.copy()
            cv2.rectangle(detection_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            axes[0, 1].imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title("Face Detection")
            axes[0, 1].axis('off')
            
            # Normalized face
            face = self.normalize_face(image, face_data)
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