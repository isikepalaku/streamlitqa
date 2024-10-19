import streamlit as st
import requests
import pandas as pd
import time
from openai import OpenAI
from typing import List, Dict

def initialize_session_state():
    """Inisialisasi session state untuk menyimpan API keys dan pengaturan."""
    # Inisialisasi API keys
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ''
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

def clean_data(text: str, openai_client, temperature: float) -> str:
    """Membersihkan data hasil scraping menggunakan OpenAI."""
    prompt = f"Bersihkan teks berikut dan buat menjadi lebih terstruktur:\n\n{text}"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Kamu adalah asisten AI yang bertugas membersihkan dan menstrukturkan data teks."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            st.error("Tidak ada respons valid dari OpenAI.")
            return text
    except Exception as e:
        st.error(f"Error saat membersihkan data: {e}")
        return text

def generate_questions(document: str, openai_client, num_questions: int = 5, temperature: float = 0.7) -> List[str]:
    """Menghasilkan pertanyaan berdasarkan dokumen."""
    prompt = f"""Berdasarkan dokumen berikut, buatlah {num_questions} pertanyaan yang mendetail dan beragam. 
    Pertanyaan-pertanyaan ini harus mencerminkan analisis hukum mendalam tentang isi dokumen:

    {document}

    Buat pertanyaan-pertanyaan yang beragam dan mendalam."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Anda adalah seorang penyidik. Tugas Anda adalah mencari dan mengidentifikasi tindak pidana, serta bertanya tentang pasal yang relevan yang dapat digunakan untuk melaporkan tindak pidana yang ditemukan berdasarkan undang-undang di Indonesia. Anda akan membuat pertanyaan yang bertujuan untuk mengonfirmasi tindak pidana dan mengidentifikasi pasal yang sesuai, termasuk bukti atau elemen yang diperlukan untuk melengkapi laporan pidana. Pastikan pertanyaan Anda membantu menelusuri apakah elemen-elemen tindak pidana terpenuhi dan bagaimana tindak pidana tersebut dapat dilaporkan."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        
        content = response.choices[0].message.content if response.choices else None
        
        if content is None:
            st.warning("Respons dari API kosong. Menggunakan pertanyaan default.")
            return [f"Pertanyaan default {i+1}" for i in range(num_questions)]
        
        questions = content.strip().split('\n')
        questions = [q.strip() for q in questions if q.strip()]
        
        return questions[:num_questions]

    except Exception as e:
        st.error(f"Error saat menghasilkan pertanyaan: {e}")
        return [f"Pertanyaan default {i+1}" for i in range(num_questions)]

def get_ai_answer(question: str, document: str, openai_client, temperature: float) -> str:
    """Mendapatkan jawaban dari OpenAI berdasarkan dokumen yang diberikan."""
    prompt = f"""Berdasarkan dokumen berikut:

    {document}

    Anda adalah penyidik kepolisian ahli hukum pidana *lex specialis* di luar KUHP, seperti UU Perlindungan Konsumen, UU Jasa Keuangan, UU Fidusia, UU Tindak Pidana Korupsi, dan UU Lingkungan Hidup. Gunakan informasi dari dokumen di atas untuk menjawab pertanyaan berikut dengan detail dan akurat, merujuk pada pasal yang relevan dan elemen hukum yang diperlukan:

    Pertanyaan: {question}
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Anda adalah penyidik kepolisian ahli hukum pidana lex specialis di luar KUHP. Tugas Anda adalah memberikan jawaban yang rinci dan akurat berdasarkan dokumen yang disediakan, merujuk pada pasal yang relevan, serta menjelaskan penerapannya dalam konteks kasus dan elemen-elemen hukum yang harus dipenuhi."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        
        content = response.choices[0].message.content if response.choices else None
        
        if content is None:
            st.warning("Respons dari API kosong. Menggunakan jawaban default.")
            return "Jawaban default"
        
        return content.strip()

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
        openai_api_key = st.text_input("Masukkan OpenAI API Key:", type="password", key="openai_key_input")
        jina_api_key = st.text_input("Masukkan Jina AI API Key:", type="password", key="jina_key_input")
        
        if st.button("Simpan API Keys"):
            st.session_state.openai_api_key = openai_api_key
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
        if not st.session_state.openai_api_key or not st.session_state.jina_api_key:
            st.error("Mohon masukkan API keys terlebih dahulu di sidebar!")
            return
        
        if not website_url:
            st.error("Mohon masukkan URL yang valid!")
            return

        try:
            # Inisialisasi OpenAI client dengan API key dari pengguna
            openai_client = OpenAI(api_key=st.session_state.openai_api_key)
            
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
                cleaned_data = clean_data(scraped_data, openai_client, st.session_state.temperature)

                # Step 3: Generating questions
                progress_text.text("Menghasilkan pertanyaan...")
                progress_bar.progress(0.6)
                questions = generate_questions(cleaned_data, openai_client, num_questions, st.session_state.temperature)

                # Step 4: Generating answers
                qa_pairs = []
                total_questions = len(questions)
                for i, question in enumerate(questions):
                    progress_text.text(f"Menghasilkan jawaban untuk pertanyaan {i+1} dari {total_questions}...")
                    progress = 0.6 + (0.4 * (i + 1) / total_questions)
                    progress_bar.progress(progress)
                    
                    answer = get_ai_answer(question, cleaned_data, openai_client, st.session_state.temperature)
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
