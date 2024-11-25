import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from langdetect import detect
from googletrans import Translator

# Fungsi untuk memuat tokenizer
def load_tokenizer(tokenizer_path='tokenizer.pkl'):
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

# Fungsi untuk memuat model
def load_model_file(model_path='model.keras'):
    model = load_model(model_path)
    return model

# Fungsi untuk menerjemahkan teks ke bahasa Inggris
def translate_to_english(text):
    lang = detect(text)
    if lang != 'en':  # Jika bahasa bukan Inggris, terjemahkan
        translator = Translator()
        translated_text = translator.translate(text, src=lang, dest='en').text
        print(f"Translated text: {translated_text}")  # Debugging print
        return translated_text
    return text

# Fungsi untuk preprocessing teks input
def preprocess_text(text, tokenizer, maxlen=200):
    # Menerjemahkan teks ke bahasa Inggris jika diperlukan
    text = translate_to_english(text)
    
    # Tokenisasi dan konversi teks ke dalam bentuk indeks kata
    new_data_tokenized = tokenizer.texts_to_sequences([text])
    
    # Padding untuk memastikan panjangnya sesuai dengan model
    new_data_padded = pad_sequences(new_data_tokenized, maxlen=maxlen, padding='post')
    
    print(f"Padded input: {new_data_padded}")  # Debugging print to check padding
    return new_data_padded

# Fungsi untuk prediksi
def predict_text(model, tokenizer, text):
    # Preprocessing dan padding
    new_data_padded = preprocess_text(text, tokenizer)
    
    # Prediksi dengan model yang sudah dilatih
    prediction = model.predict(new_data_padded)
    
    # Menentukan kelas prediksi
    predicted_class = np.argmax(prediction, axis=1)
    
    # Mengambil probabilitas untuk kelas prediksi
    predicted_prob = prediction[0][predicted_class[0]] * 100
    
    # Output prediksi dan persentase
    class_names = ['Not Suicidal', 'Suicidal']
    print(f"Predicted class: {class_names[predicted_class[0]]}")
    print(f"Prediction probability: {predicted_prob:.2f}%")
    
    return class_names[predicted_class[0]], predicted_prob
