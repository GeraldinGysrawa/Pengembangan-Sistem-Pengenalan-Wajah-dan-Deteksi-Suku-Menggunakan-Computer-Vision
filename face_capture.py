import streamlit as st
import cv2
import os
import time

def show_face_capture_page():
    """Show the face capture page"""
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