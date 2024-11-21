import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Memuat model yang sudah dilatih
model = load_model('cats_vs_dogs_model.h5')

# Judul aplikasi
st.title("Cats vs Dogs Classifier")

# Instruksi untuk pengguna
st.write("Unggah gambar kucing atau anjing untuk diklasifikasikan.")

# Upload gambar
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    # Membaca gambar menggunakan PIL
    image = Image.open(uploaded_image)
    
    # Menampilkan gambar yang diupload
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing gambar untuk prediksi
    image = image.resize((32, 32))  # Ubah ukuran gambar menjadi 32x32 sesuai dengan model
    image_array = np.array(image) / 255.0  # Normalisasi

    # Menambah dimensi batch (dari (32, 32, 3) menjadi (1, 32, 32, 3))
    image_array = np.expand_dims(image_array, axis=0)

    # Melakukan prediksi
    prediction = model.predict(image_array)
    predicted_class = (prediction > 0.5).astype(int)  # 0 untuk kucing, 1 untuk anjing

    # Menampilkan hasil prediksi
    if predicted_class == 0:
        st.write("Gambar ini adalah **Kucing**")
    else:
        st.write("Gambar ini adalah **Anjing**")