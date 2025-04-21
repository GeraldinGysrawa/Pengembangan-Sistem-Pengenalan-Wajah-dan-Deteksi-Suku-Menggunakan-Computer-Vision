import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class FaceSimilarity:
    def __init__(self):
        # Inisialisasi MTCNN untuk deteksi wajah
        self.mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Inisialisasi FaceNet untuk ekstraksi fitur
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        if torch.cuda.is_available():
            self.facenet = self.facenet.cuda()
            
        # Threshold untuk menentukan kemiripan wajah
        self.similarity_threshold = 0.7
        
    def detect_faces(self, image):
        """
        Mendeteksi wajah dalam gambar menggunakan MTCNN
        """
        # Konversi ke PIL Image jika input adalah numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        # Deteksi wajah
        boxes, probs = self.mtcnn.detect(image)
        
        if boxes is None:
            return None, None
            
        return boxes, probs
    
    def extract_face(self, image, box):
        """
        Mengekstrak region wajah dari gambar
        """
        x1, y1, x2, y2 = box
        face = image.crop((x1, y1, x2, y2))
        face = face.resize((160, 160))
        return face
    
    def get_face_embedding(self, face):
        """
        Mendapatkan embedding wajah menggunakan FaceNet
        """
        # Konversi ke tensor dan normalisasi
        face_tensor = torch.from_numpy(np.array(face)).float()
        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
        face_tensor = (face_tensor - 127.5) / 128.0
        
        if torch.cuda.is_available():
            face_tensor = face_tensor.cuda()
            
        # Ekstraksi embedding
        with torch.no_grad():
            embedding = self.facenet(face_tensor)
            
        return embedding.cpu().numpy()
    
    def compare_faces(self, embedding1, embedding2):
        """
        Membandingkan dua embedding wajah menggunakan cosine similarity
        """
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return similarity
    
    def process_image(self, image_path):
        """
        Memproses gambar dan mengembalikan embedding wajah
        """
        # Baca gambar
        image = Image.open(image_path)
        
        # Deteksi wajah
        boxes, _ = self.detect_faces(image)
        if boxes is None:
            return None
            
        # Ekstrak wajah pertama
        face = self.extract_face(image, boxes[0])
        
        # Dapatkan embedding
        embedding = self.get_face_embedding(face)
        
        return embedding
    
    def visualize_comparison(self, image1_path, image2_path, similarity_score):
        """
        Visualisasi hasil perbandingan wajah
        """
        # Baca gambar
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("Tidak dapat membaca salah satu atau kedua gambar")
        
        # Konversi BGR ke RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Deteksi wajah
        boxes1, _ = self.detect_faces(Image.fromarray(img1))
        boxes2, _ = self.detect_faces(Image.fromarray(img2))
        
        # Gambar bounding box
        if boxes1 is not None:
            x1, y1, x2, y2 = boxes1[0]
            cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
        if boxes2 is not None:
            x1, y1, x2, y2 = boxes2[0]
            cv2.rectangle(img2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Resize gambar ke ukuran yang sama
        max_height = max(img1.shape[0], img2.shape[0])
        max_width = max(img1.shape[1], img2.shape[1])
        
        # Resize gambar dengan mempertahankan aspect ratio
        def resize_with_aspect_ratio(image, max_height, max_width):
            height, width = image.shape[:2]
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height))
        
        img1_resized = resize_with_aspect_ratio(img1, max_height, max_width)
        img2_resized = resize_with_aspect_ratio(img2, max_height, max_width)
        
        # Tambahkan padding jika diperlukan
        def add_padding(image, target_height, target_width):
            height, width = image.shape[:2]
            pad_height = target_height - height
            pad_width = target_width - width
            
            # Hitung padding
            top = pad_height // 2
            bottom = pad_height - top
            left = pad_width // 2
            right = pad_width - left
            
            # Tambahkan padding dengan warna putih (255, 255, 255)
            return cv2.copyMakeBorder(image, top, bottom, left, right, 
                                    cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # Gabungkan gambar
        combined = np.hstack((
            add_padding(img1_resized, max_height, max_width),
            add_padding(img2_resized, max_height, max_width)
        ))
        
        # Tambahkan teks similarity score (dalam RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, f'Similarity: {similarity_score:.2f}', 
                   (10, 30), font, 1, (0, 200, 0), 2)
        
        return combined