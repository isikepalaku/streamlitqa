import streamlit as st
import requests
import pandas as pd
import time
from typing import List, Dict

def initialize_session_state():
    """Inisialisasi session state untuk menyimpan API keys dan pengaturan."""
    # Inisialisasi API keys
    if 'together_api_key' not in st.session_state:
        st.session_state.together_api_key = ''
    if 'jina_api_key' not in st.session_state:
        st.session_state.jina_api_key = ''
    
    # Inisialisasi pengaturan model
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    
    # Inisialisasi status proses
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = ''

def scrape_website(url: str, jina_api_key: str) -> str:
    """Melakukan scraping website menggunakan Jina AI Reader API."""
    try:
        headers = {"Authorization": f"Bearer {jina_api_key}"}
        response = requests.get(f"https://r.jina.ai/{url}", headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error saat melakukan scraping website: {e}")
        return ""

def clean_data(text: str, together_api_key: str, temperature: float) -> str:
    """Membersihkan data hasil scraping menggunakan Together.ai."""
    url = "https://api.together.xyz/v1/completions"
    
    prompt = f"Bersihkan teks berikut dan buat menjadi lebih terstruktur:\n\n{text}"
    
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": 2000,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {together_api_key}"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            cleaned_text = result['choices'][0].get('text', '').strip()
            return cleaned_text
        else:
            st.error("Tidak ada respons valid dari Together.ai")
            return text
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error saat membersihkan data dengan Together.ai: {e}")
        return text
    except Exception as e:
        st.error(f"Error tidak terduga: {e}")
        return text

def generate_questions(document: str, together_api_key: str, num_questions: int = 5, temperature: float = 0.7) -> List[str]:
    """Menghasilkan pertanyaan berdasarkan dokumen yang diberikan sebagai konteks."""
    url = "https://api.together.xyz/v1/completions"
    
    prompt = f"""Berdasarkan dokumen berikut, buatlah {num_questions} pertanyaan yang mendetail dan beragam. 
    Pertanyaan-pertanyaan ini harus mencerminkan analisis hukum mendalam dan mengacu pada informasi spesifik yang terdapat dalam dokumen:

    {document}

    Pastikan setiap pertanyaan yang dibuat mencakup upaya untuk mengidentifikasi tindak pidana, mengeksplorasi elemen-elemen hukum yang mungkin berlaku, dan mengonfirmasi pasal yang relevan serta langkah-langkah investigasi yang perlu diambil untuk melengkapi laporan pidana.
    """

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": 1000,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {together_api_key}"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0].get('text', '').strip()
            questions = content.strip().split('\n')
            questions = [q.strip() for q in questions if q.strip()]
            return questions[:num_questions]
        else:
            st.warning("Respons dari API kosong. Menggunakan pertanyaan default.")
            return [f"Pertanyaan default {i+1}" for i in range(num_questions)]

    except Exception as e:
        st.error(f"Error saat menghasilkan pertanyaan: {e}")
        return [f"Pertanyaan default {i+1}" for i in range(num_questions)]

def get_ai_answer(question: str, document: str, together_api_key: str, temperature: float) -> str:
    """Mendapatkan jawaban dari Together.ai berdasarkan dokumen yang diberikan."""
    url = "https://api.together.xyz/v1/completions"
    
    prompt = f"""Berdasarkan dokumen berikut:

    {document}

    Anda adalah penyidik kepolisian ahli hukum pidana *lex specialis* di luar KUHP, seperti UU Perlindungan Konsumen, UU Jasa Keuangan, UU Fidusia, UU Tindak Pidana Korupsi, dan UU Lingkungan Hidup. Gunakan informasi dari dokumen di atas untuk menjawab pertanyaan berikut dengan detail dan akurat, merujuk pada pasal yang relevan dan elemen hukum yang diperlukan:

    Pertanyaan: {question}
    """

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": 2000,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {together_api_key}"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0].get('text', '').strip()
        else:
            st.warning("Respons dari API kosong. Menggunakan jawaban default.")
            return "Jawaban default"

    except Exception as e:
        st.error(f"Error saat mendapatkan jawaban: {e}")
        return "Jawaban default"

def save_to_csv(qa_pairs: List[Dict], filename: str) -> str:
    """Menyimpan pertanyaan dan jawaban ke file CSV."""
    df = pd.DataFrame(qa_pairs)
    csv_file = f"{filename}.csv"
    df.to_csv(csv_file, index=False)
    return csv_file

def main():
    # Inisialisasi session state
    initialize_session_state()
    
    st.title("ZANDREGSS BOT")
    st.write("Masukkan API keys dan URL untuk menjalankan.")

    # Sidebar untuk input API keys dan pengaturan
    with st.sidebar:
        st.header("Konfigurasi")
        
        # API Keys section
        st.subheader("API Keys")
        together_api_key = st.text_input("Masukkan Together.ai API Key:", type="password", key="together_key_input")
        jina_api_key = st.text_input("Masukkan Jina AI API Key:", type="password", key="jina_key_input")
        
        if st.button("Simpan API Keys"):
            st.session_state.together_api_key = together_api_key
            st.session_state.jina_api_key = jina_api_key
            st.success("API Keys berhasil disimpan!")
        
        # Temperature setting
        st.subheader("Pengaturan Model")
        temperature = st.slider(
            "Temperatur (Kreativitas)",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Semakin tinggi nilai (mendekati 2.0), semakin kreatif dan beragam hasilnya. Semakin rendah (mendekati 0), semakin konsisten dan fokus hasilnya."
        )
        st.session_state.temperature = temperature
        
        # Explanation of temperature
        st.info("""
        **Panduan Temperatur:**
        - 0.0-0.3: Jawaban sangat konsisten dan faktual
        - 0.4-0.7: Keseimbangan antara kreativitas dan konsistensi
        - 0.8-1.2: Lebih kreatif dan beragam
        - 1.3-2.0: Sangat kreatif dan eksploratif
        """)

    # Main content
    website_url = st.text_input("Masukkan URL website:")
    num_questions = st.number_input("Berapa banyak pertanyaan yang ingin dihasilkan?", min_value=1, max_value=20, value=5)

    if st.button("Mulai Scraping"):
        if not st.session_state.together_api_key or not st.session_state.jina_api_key:
            st.error("Mohon masukkan API keys terlebih dahulu di sidebar!")
            return
        
        if not website_url:
            st.error("Mohon masukkan URL yang valid!")
            return

        try:
            # Progress container
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Step 1: Scraping
            progress_text.text("Melakukan scraping website...")
            progress_bar.progress(0.2)
            scraped_data = scrape_website(website_url, st.session_state.jina_api_key)
            
            if scraped_data:
                filename = f"scraped_data_{time.strftime('%Y%m%d-%H%M%S')}"
                
                # Step 2: Cleaning
                progress_text.text("Membersihkan data...")
                progress_bar.progress(0.4)
                cleaned_data = clean_data(scraped_data, st.session_state.together_api_key, st.session_state.temperature)

                # Step 3: Generating questions
                progress_text.text("Menghasilkan pertanyaan...")
                progress_bar.progress(0.6)
                questions = generate_questions(cleaned_data, st.session_state.together_api_key, num_questions, st.session_state.temperature)

                # Step 4: Generating answers
                qa_pairs = []
                total_questions = len(questions)
                for i, question in enumerate(questions):
                    progress_text.text(f"Menghasilkan jawaban untuk pertanyaan {i+1} dari {total_questions}...")
                    progress = 0.6 + (0.4 * (i + 1) / total_questions)
                    progress_bar.progress(progress)
                    
                    answer = get_ai_answer(question, cleaned_data, st.session_state.together_api_key, st.session_state.temperature)
                    qa_pairs.append({"Pertanyaan": question, "Jawaban": answer})

                # Clear progress indicators
                progress_text.empty()
                progress_bar.empty()

                st.success("Proses selesai!")
                
                # Menampilkan hasil dengan expander
                with st.expander("Lihat Hasil", expanded=True):
                    for i, qa in enumerate(qa_pairs, 1):
                        st.markdown(f"**Q{i}: {qa['Pertanyaan']}**")
                        st.markdown(f"A{i}: {qa['Jawaban']}")
                        st.markdown("---")
                
                # Save to CSV
                csv_file = save_to_csv(qa_pairs, filename)
                
                # Download button
                st.download_button(
                    "📥 Unduh Hasil (CSV)",
                    data=open(csv_file, "rb"),
                    file_name=f"{filename}.csv",
                    mime="text/csv",
                    help="Klik untuk mengunduh hasil dalam format CSV"
                )

        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    main()
