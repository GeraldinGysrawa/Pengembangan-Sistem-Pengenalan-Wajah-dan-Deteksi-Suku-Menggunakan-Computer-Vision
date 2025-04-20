import streamlit as st
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from ethnicity_detector import EthnicityDetector

def show_ethnicity_detection_page():
    """Show the ethnicity detection page"""
    st.title("🔍 Ethnicity Detection")
    st.markdown("""
    This feature detects the ethnicity of a person from their face image.
    """)
    
    # Add tabs for detection and training
    tab1, tab2 = st.tabs(["Detect Ethnicity", "Train Model"])
    
    with tab1:
        # Initialize ethnicity detector
        @st.cache_resource
        def load_ethnicity_detector():
            detector = EthnicityDetector(processed_dataset_path='processed_dataset')
            
            # Check for model files with different extensions
            model_files = [
                'ethnicity_model.weights.h5',
                'ethnicity_model.h5',
                'ethnicity_model.weights'
            ]
            
            model_found = False
            for model_file in model_files:
                if os.path.exists(model_file):
                    try:
                        detector.load_model(model_file)
                        st.success(f"Model loaded successfully from {model_file}!")
                        model_found = True
                        break
                    except Exception as e:
                        st.warning(f"Failed to load model from {model_file}: {str(e)}")
            
            if not model_found:
                st.warning("No trained model found. Please train a model first.")
                # Debug information
                st.info("Current directory contents:")
                st.write(os.listdir('.'))
            
            return detector
        
        detector = load_ethnicity_detector()
        
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
                
                # Detect ethnicity
                if st.button("Detect Ethnicity"):
                    if detector.model is None:
                        st.error("Model not loaded. Please train a model first.")
                    else:
                        with st.spinner("Detecting ethnicity..."):
                            try:
                                ethnicity, confidence, details = detector.predict(image)
                                
                                if ethnicity is not None:
                                    # Display main prediction
                                    st.success(f"Detected Ethnicity: {ethnicity}")
                                    st.info(f"Confidence: {confidence:.2%}")
                                    
                                    # Display face detection confidence
                                    st.info(f"Face Detection Confidence: {details['face_detection_confidence']:.2%}")
                                    
                                    # Create columns for visualization
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Display main prediction confidence bar
                                        fig1, ax1 = plt.subplots(figsize=(10, 2))
                                        ax1.barh([ethnicity], [confidence], color='skyblue')
                                        ax1.set_xlim(0, 1)
                                        ax1.set_title("Main Prediction Confidence")
                                        st.pyplot(fig1)
                                        
                                    with col2:
                                        # Display top 3 predictions
                                        fig2, ax2 = plt.subplots(figsize=(10, 2))
                                        ethnicities = [pred['ethnicity'] for pred in details['top_3_predictions']]
                                        confidences = [pred['confidence'] for pred in details['top_3_predictions']]
                                        ax2.barh(ethnicities, confidences, color=['skyblue', 'lightblue', 'lightgray'])
                                        ax2.set_xlim(0, 1)
                                        ax2.set_title("Top 3 Predictions")
                                        st.pyplot(fig2)
                                    
                                    # Display face location
                                    face_img = image.copy()
                                    x = details['face_location']['x']
                                    y = details['face_location']['y']
                                    w = details['face_location']['width']
                                    h = details['face_location']['height']
                                    cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    st.image(face_img, caption="Detected Face", use_column_width=True)
                                    
                                    # Display detailed predictions
                                    st.subheader("Detailed Predictions")
                                    for i, pred in enumerate(details['top_3_predictions'], 1):
                                        st.write(f"{i}. {pred['ethnicity']}: {pred['confidence']:.2%}")
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
                        
                        # Detect ethnicity
                        if detector.model is not None:
                            try:
                                ethnicity, confidence, details = detector.predict(frame_rgb)
                                
                                if ethnicity is not None:
                                    # Display results
                                    with result_placeholder.container():
                                        st.success(f"Detected Ethnicity: {ethnicity}")
                                        st.info(f"Confidence: {confidence:.2%}")
                                        
                                        # Display face location
                                        face_img = frame_rgb.copy()
                                        x = details['face_location']['x']
                                        y = details['face_location']['y']
                                        w = details['face_location']['width']
                                        h = details['face_location']['height']
                                        cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                        st.image(face_img, caption="Detected Face", use_column_width=True)
                                    
                                    # Display metrics
                                    with metrics_placeholder.container():
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("Main Prediction", f"{ethnicity} ({confidence:.2%})")
                                        with col2:
                                            st.metric("Face Detection", f"{details['face_detection_confidence']:.2%}")
                                        
                                        # Display top 3 predictions
                                        st.write("Top 3 Predictions:")
                                        for i, pred in enumerate(details['top_3_predictions'], 1):
                                            st.write(f"{i}. {pred['ethnicity']}: {pred['confidence']:.2%}")
                            except Exception as e:
                                st.error(f"Error during prediction: {str(e)}")
                        
                        time.sleep(0.1)  # Add small delay to reduce CPU usage
                        
                finally:
                    cap.release()
                    frame_placeholder.empty()
                    result_placeholder.empty()
                    metrics_placeholder.empty()

    with tab2:
        st.header("Train Model")
        st.markdown("""
        Train a new model using the processed dataset.
        """)
        
        # Dataset path input
        col1, col2 = st.columns(2)
        with col1:
            dataset_path = st.text_input(
                "Dataset Path",
                value="processed_dataset",
                help="Path to the dataset directory containing ethnicity folders"
            )
        
        with col2:
            model_save_path = st.text_input(
                "Model Save Path",
                value="ethnicity_model.h5",
                help="Path where the trained model will be saved"
            )
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=32, step=8)
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=20)
        with col2:
            learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
            validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.5, value=0.2, step=0.1)
        
        # Start training button
        if st.button("Start Training"):
            if not os.path.exists(dataset_path):
                st.error(f"Dataset directory '{dataset_path}' does not exist!")
            else:
                with st.spinner("Initializing training..."):
                    # Initialize detector with selected dataset
                    detector = EthnicityDetector(processed_dataset_path=dataset_path)
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create placeholders for metrics
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        train_acc_metric = st.empty()
                    with metrics_col2:
                        val_acc_metric = st.empty()
                    
                    # Create placeholder for confusion matrix
                    confusion_matrix_placeholder = st.empty()
                    
                    try:
                        # Load and preprocess data
                        status_text.text("Loading and preprocessing data...")
                        X, y = detector.load_and_preprocess_data()
                        progress_bar.progress(10)
                        
                        # Custom callback for progress updates
                        class TrainingCallback(tf.keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                # Update progress bar
                                progress = (epoch + 1) / self.params['epochs']
                                progress_bar.progress(10 + int(progress * 80))
                                
                                # Update metrics
                                train_acc = logs.get('accuracy', 0)
                                val_acc = logs.get('val_accuracy', 0)
                                train_acc_metric.metric("Training Accuracy", f"{train_acc:.2%}")
                                val_acc_metric.metric("Validation Accuracy", f"{val_acc:.2%}")
                        
                        # Train model
                        status_text.text("Training model...")
                        history = detector.train(
                            X, y,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=validation_split,
                            callbacks=[TrainingCallback()]
                        )
                        
                        # Save model
                        status_text.text("Saving model...")
                        detector.save_model(model_save_path)
                        progress_bar.progress(100)
                        
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