import streamlit as st
import requests
import pandas as pd
import time
from openai import OpenAI
from typing import List, Dict

def initialize_session_state():
    """Inisialisasi session state untuk menyimpan API keys."""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ''
    if 'jina_api_key' not in st.session_state:
        st.session_state.jina_api_key = ''

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

def generate_filename() -> str:
    """Menghasilkan nama file berdasarkan timestamp."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"scraped_data_{timestamp}"

def clean_data(text: str, openai_client) -> str:
    """Membersihkan data hasil scraping menggunakan OpenAI."""
    prompt = f"Bersihkan teks berikut dan buat menjadi lebih terstruktur:\n\n{text}"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Kamu adalah asisten AI yang bertugas membersihkan dan menstrukturkan data teks menjadi mudah dibaca manusia."},
                {"role": "user", "content": prompt}
            ]
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            st.error("Tidak ada respons valid dari OpenAI.")
            return text
    except Exception as e:
        st.error(f"Error saat membersihkan data: {e}")
        return text

def generate_questions(document: str, openai_client, num_questions: int = 5) -> List[str]:
    """Menghasilkan pertanyaan berdasarkan dokumen menggunakan OpenAI."""
    prompt = f"""Berdasarkan dokumen berikut, buatlah {num_questions} pertanyaan yang mendetail dan beragam. 
    Pertanyaan-pertanyaan ini harus mencerminkan analisis mendalam tentang isi dokumen:

    {document}

    Buat pertanyaan-pertanyaan yang beragam dan mendalam."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Anda adalah seorang penyidik. Tugas Anda adalah mencari dan mengidentifikasi tindak pidana, serta bertanya tentang pasal yang relevan yang dapat digunakan untuk melaporkan tindak pidana yang ditemukan berdasarkan undang-undang di Indonesia. Anda akan membuat pertanyaan yang bertujuan untuk mengonfirmasi tindak pidana dan mengidentifikasi pasal yang sesuai, termasuk bukti atau elemen yang diperlukan untuk melengkapi laporan pidana. Pastikan pertanyaan Anda membantu menelusuri apakah elemen-elemen tindak pidana terpenuhi dan bagaimana tindak pidana tersebut dapat dilaporkan."},
                {"role": "user", "content": prompt}
            ]
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

def get_ai_answer(question: str, document: str, openai_client) -> str:
    """Mendapatkan jawaban dari OpenAI."""
    prompt = f"""Berdasarkan dokumen berikut:

    {document}

    Jawablah pertanyaan ini dengan detail dan akurat fokus terhadap hukum pidana:
    {question}"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Anda adalah seorang penyidik kepolisian Indonesia yang sangat ahli dalam hukum pidana dan khususnya dalam penerapan pasal-pasal lex specialis dalam undang-undang. Tugas Anda adalah memberikan jawaban yang rinci dan akurat terkait penerapan pasal dalam undang-undang yang relevan. Ketika menjawab, sebutkan pasal yang berlaku, jelaskan bagaimana pasal tersebut diterapkan dalam konteks kasus yang diberikan, dan sebutkan elemen-elemen hukum yang harus dipenuhi untuk pasal tersebut dapat diterapkan. Berikan juga contoh nyata atau hipotetis yang menunjukkan bagaimana pasal tersebut telah atau dapat diterapkan."},
                {"role": "user", "content": prompt}
            ]
        )
        
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            return content.strip() if content else "Tidak ada jawaban yang dihasilkan."
        else:
            st.error("Tidak ada respons dari OpenAI.")
            return "Tidak ada jawaban yang dihasilkan."

    except Exception as e:
        st.error(f"Error saat mendapatkan jawaban AI: {e}")
        return "Terjadi kesalahan saat menghasilkan jawaban."

def save_to_csv(qa_pairs: List[Dict], filename: str) -> str:
    """Menyimpan pertanyaan dan jawaban ke file CSV tanpa header."""
    df = pd.DataFrame(qa_pairs)
    csv_file = f"{filename}.csv"
    df.to_csv(csv_file, index=False, header=False)
    return csv_file

def main():
    initialize_session_state()
    
    st.title("Web Scraper dengan Jina AI dan OpenAI")
    st.write("Masukkan API keys dan URL untuk melakukan scraping dan menghasilkan pertanyaan serta jawaban.")

    # Sidebar untuk input API keys
    with st.sidebar:
        st.header("API Key Configuration")
        openai_api_key = st.text_input("Masukkan OpenAI API Key:", type="password", key="openai_key_input")
        jina_api_key = st.text_input("Masukkan Jina AI API Key:", type="password", key="jina_key_input")
        
        if st.button("Simpan API Keys"):
            st.session_state.openai_api_key = openai_api_key
            st.session_state.jina_api_key = jina_api_key
            st.success("API Keys berhasil disimpan!")

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
            
            st.write("Melakukan scraping website...")
            scraped_data = scrape_website(website_url, st.session_state.jina_api_key)
            
            if scraped_data:
                filename = generate_filename()
                
                st.write("Membersihkan data...")
                cleaned_data = clean_data(scraped_data, openai_client)

                st.write("Menghasilkan pertanyaan dan jawaban...")
                questions = generate_questions(cleaned_data, openai_client, num_questions)

                qa_pairs = []
                for question in questions:
                    answer = get_ai_answer(question, cleaned_data, openai_client)
                    qa_pairs.append({"Pertanyaan": question, "Jawaban": answer})

                csv_file = save_to_csv(qa_pairs, filename)
                st.success("Proses selesai!")
                
                # Menampilkan hasil
                st.subheader("Hasil Scraping:")
                for i, qa in enumerate(qa_pairs, 1):
                    st.write(f"Q{i}: {qa['Pertanyaan']}")
                    st.write(f"A{i}: {qa['Jawaban']}")
                    st.write("---")
                
                st.download_button(
                    "Unduh CSV", 
                    data=open(csv_file, "rb"), 
                    file_name=csv_file, 
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    main()
