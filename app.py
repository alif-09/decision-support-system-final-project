import streamlit as st
from utils import load_tokenizer, load_model_file, predict_text, make_rounded_image
import os

# Load resources
tokenizer = load_tokenizer('tokenizer.pkl')
model = load_model_file('best_model.keras')

# Page Configuration
st.set_page_config(
    page_title="Prediksi Kondisi Mental",
    page_icon="🧠",
    layout="centered"
)

# App Title and Description
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50; font-size: 36px;'>🧠 Aplikasi Prediksi Kondisi Mental</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align: center;">
        Selamat datang di <strong>Aplikasi Prediksi Kondisi Mental</strong>!  
        Aplikasi ini dirancang untuk membantu mendeteksi apakah teks yang Anda masukkan menunjukkan tanda-tanda suicidal atau tidak.  
        <br><br>
        <strong>Harap masukkan deskripsi perasaan Anda di bawah ini.</strong>
    </div>
    """,
    unsafe_allow_html=True
)

# Input Section
st.markdown("---")
st.header("💬 Masukkan Teks Anda")
text = st.text_area(
    "Ceritakan perasaan Anda di sini:",
    placeholder="Contoh: Saya merasa sangat sedih dan tidak tahu harus berbuat apa...",
    height=150
)

# Prediction Button
if st.button("🔍 Prediksi"):
    if not text.strip():
        st.error("❌ Input tidak valid. Mohon masukkan teks yang tidak kosong.")
    else:
        with st.spinner("⏳ Memproses prediksi..."):
            try:
                # Simulate loading time for animation (optional, can be removed in production)
                import time
                time.sleep(2)

                # Get prediction and probability
                predicted_class, predicted_prob = predict_text(model, tokenizer, text)

                # Display Prediction Results
                st.markdown("---")
                st.header("📊 Hasil Prediksi")

                st.markdown(
                    f"""
                    **Teks Anda:**  
                    <blockquote style="font-style: italic; color: #555;">{text.strip()}</blockquote>
                    """,
                    unsafe_allow_html=True
                )

                if predicted_class == "Suicidal":
                    st.error("⚠️ **Hasil Prediksi: Anda butuh penanganan.**")
                    st.markdown(
                        f"""
                        **Kategori:** {predicted_class}  
                        **Kepercayaan Model:** {predicted_prob:.2f}%  
                        > **Pesan Penting:**  
                        > Mohon segera mencari bantuan profesional jika Anda merasa tidak aman. Anda tidak sendiri, dan ada orang-orang yang peduli pada Anda.
                        """
                    )
                else:
                    st.success("✅ **Hasil Prediksi: Anda tampak baik-baik saja. Tetap jaga kondisi Anda.**")
                    st.markdown(
                        f"""
                        **Kategori:** {predicted_class}  
                        **Kepercayaan Model:** {predicted_prob:.2f}%  
                        > **Pesan Penting:**  
                        > Tetap jaga kesehatan mental Anda dan hubungi seseorang jika Anda membutuhkan dukungan.
                        """
                    )
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")


# Footer Section
st.markdown("---")
st.markdown(
    """
    💡 _**Catatan:** Hasil prediksi ini hanya bersifat indikatif dan bukan pengganti bantuan profesional.  
    Jika Anda merasa kesulitan, jangan ragu untuk mencari bantuan._  
    """
)

# Load and display photos of developers from public folder
st.markdown("---")  
st.markdown("""<div style="text-align: center;"><h4>Tim Pengembang</h4></div>""",
            unsafe_allow_html=True)
cols = st.columns(3)
with cols[0]:
    st.image(make_rounded_image('public/zhafran.jpg'), width=150,
             caption="Muhammad Zafran Shiddiq\n140810220007")
with cols[1]:
    st.image(make_rounded_image('public/tegar.jpg'), width=150,
             caption="Tegar Muhamad Rizki\n140810220034")
with cols[2]:
    st.image(make_rounded_image('public/alif.jpg'), width=150,
             caption="Alif Al Husaini\n140810220036")