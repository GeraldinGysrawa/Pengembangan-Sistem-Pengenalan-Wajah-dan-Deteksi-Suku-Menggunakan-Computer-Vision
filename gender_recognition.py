import streamlit as st
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
from mtcnn import MTCNN
import seaborn as sns
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenderRecognizer:
    def __init__(self, model_path=None, img_size=(224, 224)):
        logger.info("Initializing GenderRecognizer")
        self.img_size = img_size
        self.detector = MTCNN()
        self.model = None
        self.classes = ['Female', 'Male']  # Fixed for gender recognition
        
        # Build model architecture
        self._build_model()
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _build_model(self):
        """Build MobileNetV2-based model for gender classification"""
        logger.info("Building gender recognition model")
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(self.classes), activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("Model built successfully")
        return model
    
    def train_from_csv(self, csv_path, batch_size=32, epochs=10, validation_split=0.2, callbacks=None):
        """Train the model using data from CSV file"""
        logger.info(f"Training model from CSV: {csv_path}")
        
        # Load and preprocess data from CSV
        df = pd.read_csv(csv_path)
        
        # Create data generators
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        
        # Split dataframe
        train_df, val_df = train_test_split(df, test_size=validation_split, stratify=df['gender'], random_state=42)
        
        # Save split dataframes temporarily for generators
        train_df.to_csv('temp_train.csv', index=False)
        val_df.to_csv('temp_val.csv', index=False)
        
        # Create generators
        train_generator = datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='image_path',
            y_col='gender',
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col='image_path',
            y_col='gender',
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Unfreeze some layers for fine-tuning
        for layer in self.model.layers[-20:]:
            layer.trainable = True
        
        # Recompile model with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=callbacks
        )
        
        # Clean up temp files
        os.remove('temp_train.csv')
        os.remove('temp_val.csv')
        
        logger.info("Training completed")
        return history
    
    def detect_face(self, image):
        """Detect face using MTCNN"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Detect faces with MTCNN
            faces = self.detector.detect_faces(image)
            
            if not faces:
                logger.warning("No faces detected in image")
                return None
            
            # Get the face with highest confidence
            face = max(faces, key=lambda x: x['confidence'])
            
            # Check confidence threshold
            if face['confidence'] < 0.9:
                logger.warning(f"Face detection confidence too low: {face['confidence']:.2f}")
                return None
            
            # Extract face box
            x, y, w, h = face['box']
            
            # Add padding to face detection
            padding = int(0.1 * max(w, h))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            face_img = image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, self.img_size)
            
            return {
                'face': face_img,
                'box': (x, y, w, h),
                'confidence': face['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return None
    
    def predict(self, image):
        """Predict gender from an image"""
        logger.info("Making gender prediction")
        if self.model is None:
            logger.error("Model not loaded. Please load the model first.")
            return None, None, None
            
        # Detect face
        face_data = self.detect_face(image)
        if face_data is None:
            logger.warning("No face detected in image")
            return None, None, None
        
        # Preprocess face for prediction
        face_img = face_data['face'] / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        
        try:
            # Get prediction
            predictions = self.model.predict(face_img, verbose=0)
            predicted_class = np.argmax(predictions[0])
            prediction_confidence = predictions[0][predicted_class]
            
            # Get gender label
            gender = self.classes[predicted_class]
            
            # Create prediction details
            prediction_details = {
                'gender': gender,
                'confidence': float(prediction_confidence),
                'face_detection_confidence': float(face_data['confidence']),
                'face_location': {
                    'x': int(face_data['box'][0]),
                    'y': int(face_data['box'][1]),
                    'width': int(face_data['box'][2]),
                    'height': int(face_data['box'][3])
                },
                'class_probabilities': {
                    'Female': float(predictions[0][0]),
                    'Male': float(predictions[0][1])
                }
            }
            
            logger.info(f"Predicted gender: {gender} with confidence: {prediction_confidence:.4f}")
            
            return gender, prediction_confidence, prediction_details
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None, None, None
    
    def save_model(self, model_path):
        """Save model weights"""
        logger.info(f"Saving model to {model_path}")
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
    def load_model(self, model_path):
        """Load model weights"""
        logger.info(f"Loading model from {model_path}")
        try:
            self.model.load_weights(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            try:
                # Try to load as full model
                self.model = tf.keras.models.load_model(model_path)
                logger.info(f"Full model loaded successfully from {model_path}")
            except Exception as e2:
                logger.error(f"Failed to load model: {str(e2)}")
                raise ValueError(f"Could not load model from {model_path}: {str(e2)}")

def show_gender_recognition_page():
    """Show the gender recognition page"""
    st.title("👫 Gender Recognition")
    st.markdown("""
    This feature recognizes the gender of a person from their face image using computer vision and deep learning.
    """)
    
    # Add tabs for detection and training
    tab1, tab2 = st.tabs(["Detect Gender", "Train Model"])
    
    with tab1:
        # Initialize gender recognizer
        @st.cache_resource
        def load_gender_recognizer():
            # Look for pre-trained model
            model_files = [
                'gender_model.h5',
                'gender_model.weights.h5',
                'gender_model'
            ]
            
            recognizer = GenderRecognizer()
            model_found = False
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    try:
                        recognizer.load_model(model_file)
                        st.success(f"Model loaded successfully from {model_file}!")
                        model_found = True
                        break
                    except Exception as e:
                        st.warning(f"Failed to load model from {model_file}: {str(e)}")
            
            if not model_found:
                st.warning("No trained gender model found. Using default model with pre-trained MobileNetV2 weights.")
                # Initialize with default model only (ImageNet weights)
            
            return recognizer
        
        recognizer = load_gender_recognizer()
        
        # Input source selection
        input_source = st.radio(
            "Select Input Source",
            ["Upload Image", "Webcam"],
            horizontal=True
        )
        
        if input_source == "Upload Image":
            # File uploader
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Read image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Display image
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Detect gender
                if st.button("Detect Gender"):
                    with st.spinner("Detecting gender..."):
                        try:
                            gender, confidence, details = recognizer.predict(image)
                            
                            if gender is not None:
                                # Display main prediction
                                st.success(f"Detected Gender: {gender}")
                                st.info(f"Confidence: {confidence:.2%}")
                                
                                # Display face detection confidence
                                st.info(f"Face Detection Confidence: {details['face_detection_confidence']:.2%}")
                                
                                # Create columns for visualization
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Display gender confidence bar
                                    fig1, ax1 = plt.subplots(figsize=(10, 2))
                                    ax1.barh([gender], [confidence], color='skyblue')
                                    ax1.set_xlim(0, 1)
                                    ax1.set_title("Gender Prediction Confidence")
                                    st.pyplot(fig1)
                                    
                                with col2:
                                    # Display class probabilities
                                    fig2, ax2 = plt.subplots(figsize=(10, 2))
                                    classes = ['Female', 'Male']
                                    probs = [details['class_probabilities']['Female'], 
                                             details['class_probabilities']['Male']]
                                    ax2.barh(classes, probs, color=['pink', 'lightblue'])
                                    ax2.set_xlim(0, 1)
                                    ax2.set_title("Gender Probabilities")
                                    st.pyplot(fig2)
                                
                                # Display face location
                                face_img = image.copy()
                                x = details['face_location']['x']
                                y = details['face_location']['y']
                                w = details['face_location']['width']
                                h = details['face_location']['height']
                                
                                # Draw gender-specific colored rectangle
                                color = (255, 105, 180) if gender == "Female" else (0, 191, 255)  # pink for female, blue for male
                                cv2.rectangle(face_img, (x, y), (x+w, y+h), color, 3)
                                
                                # Add gender label with confidence
                                label = f"{gender}: {confidence:.2%}"
                                cv2.putText(face_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                                
                                st.image(face_img, caption="Detected Gender", use_column_width=True)
                                
                            else:
                                st.error("No face detected in the image. Please try another image.")
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
        
        else:  # Webcam
            st.write("Click 'Start Webcam' to begin detection")
            
            # Initialize webcam
            if 'webcam_running' not in st.session_state:
                st.session_state.webcam_running = False
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Webcam"):
                    st.session_state.webcam_running = True
            with col2:
                if st.button("Stop Webcam"):
                    st.session_state.webcam_running = False
            
            # Webcam feed
            if st.session_state.webcam_running:
                cap = cv2.VideoCapture(0)
                
                # Create placeholders for webcam feed and results
                frame_placeholder = st.empty()
                result_placeholder = st.empty()
                metrics_placeholder = st.empty()
                
                try:
                    while st.session_state.webcam_running:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to access webcam")
                            break
                        
                        # Convert frame to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame
                        frame_placeholder.image(frame_rgb, channels="RGB")
                        
                        # Detect gender
                        if recognizer.model is not None:
                            try:
                                gender, confidence, details = recognizer.predict(frame_rgb)
                                
                                if gender is not None:
                                    # Create output frame with detection
                                    output_frame = frame_rgb.copy()
                                    x = details['face_location']['x']
                                    y = details['face_location']['y']
                                    w = details['face_location']['width']
                                    h = details['face_location']['height']
                                    
                                    # Gender-specific color
                                    color = (255, 105, 180) if gender == "Female" else (0, 191, 255)
                                    
                                    # Draw rectangle and label
                                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 3)
                                    label = f"{gender}: {confidence:.2%}"
                                    cv2.putText(output_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                                    
                                    # Display results
                                    frame_placeholder.image(output_frame, channels="RGB")
                                    
                                    # Display metrics
                                    with metrics_placeholder.container():
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("Gender", f"{gender} ({confidence:.2%})")
                                        with col2:
                                            st.metric("Face Detection", f"{details['face_detection_confidence']:.2%}")
                                
                            except Exception as e:
                                st.error(f"Error during prediction: {str(e)}")
                        
                        time.sleep(0.1)  # Add small delay to reduce CPU usage
                        
                finally:
                    cap.release()
                    frame_placeholder.empty()
                    result_placeholder.empty()
                    metrics_placeholder.empty()

    with tab2:
        st.header("Train Gender Recognition Model")
        st.markdown("""
        Train a new gender recognition model using your own dataset.
        """)
        
        # Dataset path input
        col1, col2 = st.columns(2)
        with col1:
            csv_path = st.text_input(
                "CSV Dataset Path",
                value="dataset_gender.csv",
                help="Path to the CSV file containing gender dataset information"
            )
        
        with col2:
            model_save_path = st.text_input(
                "Model Save Path",
                value="gender_model.h5",
                help="Path where the trained model will be saved"
            )
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=32, step=8)
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
        with col2:
            learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
            validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.5, value=0.2, step=0.1)
        
        # Start training button
        if st.button("Start Training"):
            if not os.path.exists(csv_path):
                st.error(f"CSV file '{csv_path}' does not exist!")
            else:
                with st.spinner("Initializing training..."):
                    # Initialize recognizer
                    recognizer = GenderRecognizer()
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create placeholders for metrics
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        train_acc_metric = st.empty()
                    with metrics_col2:
                        val_acc_metric = st.empty()
                    
                    try:
                        # Custom callback for progress updates
                        class TrainingCallback(tf.keras.callbacks.Callback):
                            def on_epoch_begin(self, epoch, logs=None):
                                status_text.text(f"Training epoch {epoch+1}/{epochs}...")
                                
                            def on_epoch_end(self, epoch, logs=None):
                                # Update progress bar
                                progress = (epoch + 1) / epochs
                                progress_bar.progress(progress)
                                
                                # Update metrics
                                train_acc = logs.get('accuracy', 0)
                                val_acc = logs.get('val_accuracy', 0)
                                train_acc_metric.metric("Training Accuracy", f"{train_acc:.2%}")
                                val_acc_metric.metric("Validation Accuracy", f"{val_acc:.2%}")
                        
                        # Train model
                        status_text.text("Training model...")
                        history = recognizer.train_from_csv(
                            csv_path=csv_path,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=validation_split,
                            callbacks=[TrainingCallback()]
                        )
                        
                        # Save model
                        status_text.text("Saving model...")
                        recognizer.save_model(model_save_path)
                        progress_bar.progress(1.0)
                        
                        # Show final results
                        status_text.text("Training completed!")
                        st.success("Model trained and saved successfully!")
                        
                        # Plot training history
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                        
                        # Plot accuracy
                        ax1.plot(history.history['accuracy'], label='Training')
                        ax1.plot(history.history['val_accuracy'], label='Validation')
                        ax1.set_title('Model Accuracy')
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('Accuracy')
                        ax1.legend()
                        
                        # Plot loss
                        ax2.plot(history.history['loss'], label='Training')
                        ax2.plot(history.history['val_loss'], label='Validation')
                        ax2.set_title('Model Loss')
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('Loss')
                        ax2.legend()
                        
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"An error occurred during training: {str(e)}")
                        progress_bar.progress(0)
                        status_text.text("Training failed!")
        
        # Dataset statistics
        if os.path.exists(csv_path):
            st.subheader("Dataset Statistics")
            try:
                df = pd.read_csv(csv_path)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Gender distribution
                    gender_counts = df['gender'].value_counts()
                    st.metric("Total Images", len(df))
                    
                with col2:
                    # Number of male images
                    male_count = gender_counts.get('Male', 0)
                    st.metric("Male Images", male_count)
                    
                with col3:
                    # Number of female images
                    female_count = gender_counts.get('Female', 0)
                    st.metric("Female Images", female_count)
                
                # Gender distribution plot
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(x='gender', data=df, palette=['pink', 'lightblue'])
                ax.set_title('Gender Distribution in Dataset')
                st.pyplot(fig)
                
                # Ethnicity distribution
                if 'ethnicity' in df.columns:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ethnicity_gender = pd.crosstab(df['ethnicity'], df['gender'])
                    ethnicity_gender.plot(kind='bar', stacked=True, ax=ax, color=['pink', 'lightblue'])
                    ax.set_title('Gender Distribution by Ethnicity')
                    ax.set_xlabel('Ethnicity')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error analyzing dataset: {str(e)}")
                
        # Instructions
        st.header("Instructions")
        st.markdown("""
        1. Make sure your CSV file has the following columns: 'name', 'gender', and 'image_path'
        2. The gender column should contain 'Male' or 'Female' values
        3. The image_path column should contain valid paths to the face images
        4. Adjust the training parameters according to your dataset size
        5. Click 'Start Training' to begin the training process
        """) 