import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EthnicityDetector:
    def __init__(self, processed_dataset_path='processed_dataset', img_size=(224, 224)):
        logger.info(f"Initializing EthnicityDetector with processed_dataset_path={processed_dataset_path}")
        self.processed_dataset_path = processed_dataset_path
        self.img_size = img_size
        self.detector = MTCNN()
        self.label_encoder = LabelEncoder()
        self.model = None
        
        # Initialize model if processed dataset exists
        if os.path.exists(processed_dataset_path):
            ethnicities = [d for d in os.listdir(processed_dataset_path) 
                         if os.path.isdir(os.path.join(processed_dataset_path, d))]
            if ethnicities:
                self.label_encoder.fit(ethnicities)
                num_classes = len(ethnicities)
                self.model = self.build_model(num_classes)
                self.model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("Model initialized with classes: " + ", ".join(ethnicities))
    
    def load_and_preprocess_data(self):
        """Load preprocessed data from processed_dataset"""
        logger.info("Loading data from processed dataset...")
        
        # Get list of ethnicities from folder names
        ethnicities = [d for d in os.listdir(self.processed_dataset_path) 
                      if os.path.isdir(os.path.join(self.processed_dataset_path, d))]
        
        if not ethnicities:
            raise ValueError(f"No ethnicity folders found in {self.processed_dataset_path}")
            
        logger.info(f"Found ethnicities: {ethnicities}")
        self.label_encoder.fit(ethnicities)
        
        # Prepare data
        X = []
        y = []
        
        logger.info("Loading preprocessed images...")
        for ethnicity in ethnicities:
            ethnicity_path = os.path.join(self.processed_dataset_path, ethnicity)
            
            # Get all person folders
            person_folders = [d for d in os.listdir(ethnicity_path) 
                            if os.path.isdir(os.path.join(ethnicity_path, d))]
            
            for person_folder in person_folders:
                person_path = os.path.join(ethnicity_path, person_folder)
                image_files = [f for f in os.listdir(person_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if not image_files:
                    logger.warning(f"No images found in {person_path}")
                    continue
                    
                for img_file in tqdm(image_files, desc=f"Loading {ethnicity}/{person_folder} images"):
                    img_path = os.path.join(person_path, img_file)
                    try:
                        # Read image (already preprocessed)
                        img = cv2.imread(img_path)
                        if img is None:
                            logger.warning(f"Failed to read image: {img_path}")
                            continue
                            
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Normalize
                        img = img / 255.0
                        
                        X.append(img)
                        y.append(ethnicity)
                    except Exception as e:
                        logger.error(f"Error processing image {img_path}: {str(e)}")
        
        if not X:
            raise ValueError("No valid images were processed")
            
        X = np.array(X)
        y = self.label_encoder.transform(y)
        
        logger.info(f"Processed {len(X)} images successfully")
        logger.info(f"Final data shapes - X: {X.shape}, y: {y.shape}")
        
        return X, y
    
    def build_model(self, num_classes):
        """Build ResNet50-based model for ethnicity classification"""
        logger.info(f"Building model with {num_classes} classes")
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        logger.info("Model built successfully")
        return model
    
    def train(self, X, y, batch_size=32, epochs=20, validation_split=0.2, callbacks=None):
        """Train the model"""
        logger.info("Starting model training...")
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        logger.info(f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}")
        
        # Build model
        num_classes = len(self.label_encoder.classes_)
        self.model = self.build_model(num_classes)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        import seaborn as sns
        
        # Classification report
        class_report = classification_report(y_test, y_pred_classes, 
                                          target_names=self.label_encoder.classes_,
                                          output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # ROC curves for each class
        n_classes = len(self.label_encoder.classes_)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Create evaluation results dictionary
        eval_results = {
            'classification_report': class_report,
            'confusion_matrix': cm,
            'roc_curves': {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }
        }
        
        # Visualize results
        self._visualize_evaluation_results(eval_results)
        
        return eval_results
    
    def _visualize_evaluation_results(self, eval_results):
        """Visualize evaluation results"""
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(eval_results['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        for i in range(len(self.label_encoder.classes_)):
            plt.plot(eval_results['roc_curves']['fpr'][i], 
                    eval_results['roc_curves']['tpr'][i],
                    label=f'{self.label_encoder.classes_[i]} (AUC = {eval_results["roc_curves"]["auc"][i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Ethnicity')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_curves.png')
        plt.close()
        
        # Save classification report
        report_df = pd.DataFrame(eval_results['classification_report']).transpose()
        report_df.to_csv('classification_report.csv')
        
        logger.info("Evaluation visualizations saved to disk")
    
    def predict(self, image):
        """Predict ethnicity from an image"""
        logger.info("Making prediction...")
        if self.model is None:
            logger.error("Model not loaded. Please load the model first.")
            return None, None, None
            
        # Preprocess image
        faces = self.detector.detect_faces(image)
        if not faces:
            logger.warning("No face detected in image")
            return None, None, None
        
        # Get the largest face (assuming it's the main subject)
        face = max(faces, key=lambda x: x['confidence'])
        x, y, w, h = face['box']
        confidence = face['confidence']
        
        # Add padding to face detection
        padding = int(0.1 * max(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        
        # Extract and preprocess face
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, self.img_size)
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        
        try:
            # Get prediction
            predictions = self.model.predict(face_img, verbose=0)
            predicted_class = np.argmax(predictions[0])
            prediction_confidence = predictions[0][predicted_class]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            top_3_ethnicities = self.label_encoder.inverse_transform(top_3_idx)
            top_3_confidences = predictions[0][top_3_idx]
            
            # Main prediction
            ethnicity = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Create prediction details
            prediction_details = {
                'main_prediction': {
                    'ethnicity': ethnicity,
                    'confidence': float(prediction_confidence)
                },
                'top_3_predictions': [
                    {'ethnicity': eth, 'confidence': float(conf)}
                    for eth, conf in zip(top_3_ethnicities, top_3_confidences)
                ],
                'face_detection_confidence': float(confidence),
                'face_location': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                }
            }
            
            logger.info(f"Predicted ethnicity: {ethnicity} with confidence: {prediction_confidence:.4f}")
            logger.info(f"Face detection confidence: {confidence:.4f}")
            
            return ethnicity, prediction_confidence, prediction_details
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None, None, None
    
    def save_model(self, model_path):
        """Save model weights"""
        logger.info(f"Saving model to {model_path}")
        if not model_path.endswith('.weights.h5'):
            model_path = model_path.replace('.h5', '.weights.h5')
            if not model_path.endswith('.weights.h5'):
                model_path = model_path + '.weights.h5'
        self.model.save_weights(model_path)
        logger.info(f"Model weights saved to {model_path}")
        
    def load_model(self, model_path):
        """Load model weights"""
        logger.info(f"Loading model from {model_path}")
        
        # Try different model file extensions
        possible_paths = [
            model_path,
            model_path.replace('.h5', '.weights.h5'),
            model_path + '.weights.h5',
            model_path.replace('.weights.h5', '.h5')
        ]
        
        model_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    if self.model is None:
                        # Get number of classes from processed dataset
                        ethnicities = [d for d in os.listdir(self.processed_dataset_path) 
                                    if os.path.isdir(os.path.join(self.processed_dataset_path, d))]
                        if not ethnicities:
                            raise ValueError(f"No ethnicity folders found in {self.processed_dataset_path}")
                            
                        self.label_encoder.fit(ethnicities)
                        num_classes = len(ethnicities)
                        self.model = self.build_model(num_classes)
                        
                        # Compile model before loading weights
                        self.model.compile(
                            optimizer=Adam(learning_rate=0.001),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                        )
                    
                    self.model.load_weights(path)
                    logger.info(f"Model weights loaded from {path}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model from {path}: {str(e)}")
        
        if not model_loaded:
            error_msg = f"Could not load model from any of these paths: {possible_paths}"
            logger.error(error_msg)
            raise ValueError(error_msg) 