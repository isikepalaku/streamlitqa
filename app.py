import streamlit as st
import requests
import pandas as pd
import os
import re
import time
from openai import OpenAI
from typing import List, Dict

# Inisialisasi klien OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Pastikan sudah diatur di environment
jina_api_key = os.getenv("JINA_API_KEY")  # Pastikan sudah diatur di environment

jina_headers = {
    "Authorization": f"Bearer {jina_api_key}"
}

def scrape_website(url: str) -> str:
    """Melakukan scraping website menggunakan Jina AI Reader API."""
    try:
        response = requests.get(f"https://r.jina.ai/{url}", headers=jina_headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error saat melakukan scraping website: {e}")
        return ""

def generate_filename() -> str:
    """Menghasilkan nama file berdasarkan timestamp."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"scraped_data_{timestamp}"

def clean_data(text: str) -> str:
    """Membersihkan data hasil scraping menggunakan OpenAI."""
    prompt = f"Bersihkan teks berikut dan buat menjadi lebih terstruktur:\n\n{text}"
    try:
        response = client.chat.completions.create(
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

def generate_questions(document: str, num_questions: int = 5) -> List[str]:
    """Menghasilkan pertanyaan berdasarkan dokumen menggunakan OpenAI."""
    prompt = f"""Berdasarkan dokumen berikut, buatlah {num_questions} pertanyaan yang mendetail dan beragam. 
    Pertanyaan-pertanyaan ini harus mencerminkan analisis mendalam tentang isi dokumen:

    {document}

    Buat pertanyaan-pertanyaan yang beragam dan mendalam."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Anda adalah ahli hukum berpengalaman yang membuat pertanyaan dari konteks yang diberikan"},
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

def get_ai_answer(question: str, document: str) -> str:
    """Mendapatkan jawaban dari OpenAI."""
    prompt = f"""Berdasarkan dokumen berikut:

    {document}

    Jawablah pertanyaan ini dengan detail dan akurat:
    {question}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Buat pertanyaan dari konteks kaitkan dengan hukum pidana."},
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
    df.to_csv(csv_file, index=False, header=False)  # Menghilangkan header
    return csv_file

def main():
    st.title("Web Scraper dengan Jina AI dan OpenAI")
    st.write("Masukkan URL untuk melakukan scraping dan menghasilkan pertanyaan serta jawaban.")

    website_url = st.text_input("Masukkan URL website:")
    num_questions = st.number_input("Berapa banyak pertanyaan yang ingin dihasilkan?", min_value=1, max_value=20, value=5)

    if st.button("Mulai Scraping"):
        if website_url:
            st.write("Melakukan scraping website...")
            scraped_data = scrape_website(website_url)
            
            if scraped_data:
                # Menghasilkan nama file berdasarkan waktu
                filename = generate_filename()
                
                st.write("Membersihkan data...")
                cleaned_data = clean_data(scraped_data)

                st.write("Menghasilkan pertanyaan dan jawaban...")
                questions = generate_questions(cleaned_data, num_questions)

                qa_pairs = []
                for question in questions:
                    answer = get_ai_answer(question, cleaned_data)
                    qa_pairs.append({"Pertanyaan": question, "Jawaban": answer})

                csv_file = save_to_csv(qa_pairs, filename)
                st.success("Proses selesai!")
                st.download_button("Unduh CSV", data=open(csv_file, "rb"), file_name=csv_file, mime="text/csv")
        else:
            st.error("Mohon masukkan URL yang valid.")

if __name__ == "__main__":
    main()