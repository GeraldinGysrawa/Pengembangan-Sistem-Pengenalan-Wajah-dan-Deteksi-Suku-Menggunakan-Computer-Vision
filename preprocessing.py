import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime
import random
import matplotlib.pyplot as plt

class FacePreprocessor:
    """
    A class to handle face preprocessing operations including detection, normalization, and augmentation.
    """
    
    def __init__(self, input_dataset_path, output_dataset_path, metadata_path):
        """
        Initialize the preprocessor with input and output paths.
        
        Args:
            input_dataset_path (str): Path to the input dataset
            output_dataset_path (str): Path to save the processed dataset
            metadata_path (str): Path to the metadata CSV file
        """
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
        """
        Detect face in the image and return the bounding box.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: (x, y, w, h) or None if no face detected
        """
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
        """
        Extract and normalize the face region.
        
        Args:
            image (numpy.ndarray): Input image
            face_bbox (tuple): Bounding box of the face (x, y, w, h)
            target_size (tuple): Target size (width, height)
            
        Returns:
            numpy.ndarray: Normalized face image
        """
        x, y, w, h = face_bbox
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        return face
    
    def augment_image(self, image, do_flip=True, do_rotate=True, 
                     do_brightness=True, do_noise=True):
        """
        Apply various augmentation techniques to the image.
        
        Args:
            image (numpy.ndarray): Input image
            do_flip (bool): Whether to apply horizontal flip
            do_rotate (bool): Whether to apply random rotation
            do_brightness (bool): Whether to apply brightness adjustment
            do_noise (bool): Whether to apply Gaussian noise
            
        Returns:
            list: List of augmented images
        """
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
    
    def process_image(self, image_path, output_path, person_name, ethnicity, expression="normal", 
                     angle="frontal", lighting="normal"):
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path (str): Path to the input image
            output_path (str): Path to save the processed image
            person_name (str): Name of the person
            ethnicity (str): Ethnicity of the person
            expression (str): Expression in the image
            angle (str): Angle of the face
            lighting (str): Lighting condition
            
        Returns:
            list: List of processed image paths
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error reading image: {image_path}")
            return []
        
        # Detect face
        face_bbox = self.detect_face(image)
        if face_bbox is None:
            print(f"No face detected in: {image_path}")
            return []
        
        # Normalize face
        normalized_face = self.normalize_face(image, face_bbox)
        
        # Augment face
        augmented_faces = self.augment_image(normalized_face)
        
        # Save processed images
        processed_paths = []
        for i, aug_face in enumerate(augmented_faces):
            # Create output directory structure
            os.makedirs(output_path, exist_ok=True)
            
            # Generate output filename
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            
            if i == 0:
                # Original processed image
                output_filename = f"{name}_processed{ext}"
            elif i == 1:
                # Flipped
                output_filename = f"{name}_flipped{ext}"
            elif i == 2:
                # Rotated
                output_filename = f"{name}_rotated{ext}"
            elif i == 3:
                # Brightness adjusted
                output_filename = f"{name}_brightness{ext}"
            else:
                # Noisy
                output_filename = f"{name}_noisy{ext}"
            
            output_filepath = os.path.join(output_path, output_filename)
            cv2.imwrite(output_filepath, aug_face)
            processed_paths.append(output_filepath)
        
        return processed_paths
    
    def process_dataset(self, target_size=(224, 224), do_flip=True, 
                       do_rotate=True, do_brightness=True, do_noise=True):
        """
        Process the entire dataset and apply augmentations.
        
        Returns:
            dict: Statistics about the processing
        """
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
    
    def visualize_processing(self, image_path):
        """
        Visualize the processing steps for a given image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            matplotlib.figure.Figure: Figure with visualization
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Could not read image")
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
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
            
            # Augmentations
            augmented_faces = self.augment_image(face)
            
            # Horizontal flip
            axes[1, 0].imshow(cv2.cvtColor(augmented_faces[1], cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title("Horizontal Flip")
            axes[1, 0].axis('off')
            
            # Rotation
            axes[1, 1].imshow(cv2.cvtColor(augmented_faces[2], cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title("Random Rotation")
            axes[1, 1].axis('off')
            
            # Brightness
            axes[1, 2].imshow(cv2.cvtColor(augmented_faces[3], cv2.COLOR_BGR2RGB))
            axes[1, 2].set_title("Brightness Adjustment")
            axes[1, 2].axis('off')
        else:
            for i in range(1, 6):
                ax = axes[i//3, i%3]
                ax.text(0.5, 0.5, "No face detected", 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes)
                ax.axis('off')
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = FacePreprocessor(
        input_dataset_path="dataset_tubes",
        output_dataset_path="processed_dataset",
        metadata_path="dataset_tubes/metadata.csv"
    )
    
    # Process dataset
    stats = preprocessor.process_dataset()
    
    # Print statistics
    print(f"Total images: {stats['total_images']}")
    print(f"Processed images: {stats['processed_images']}")
    print(f"Failed images: {stats['failed_images']}")
    print(f"Augmented images: {stats['augmented_images']}")
    print(f"People: {len(stats['people'])}")
    print(f"Ethnicities: {len(stats['ethnicities'])}") 