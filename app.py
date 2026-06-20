import streamlit as st
import pandas as pd
import docx
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Evaluasi Tesis - Magister Teknik Informatika", layout="wide")

st.title("🎓 Sistem Evaluasi Cerdas Usulan Judul Tesis")
st.subheader("Program Studi Magister Teknik Informatika - Universitas Pamulang")
st.markdown("Sistem ini mengevaluasi kebaruan (novelty) usulan judul tesis menggunakan pendekatan NLP (Sentence-BERT) dan Generative AI (Gemini).")
st.divider()

# --- SIDEBAR: KONFIGURASI API KEY ---
st.sidebar.header("🔑 Konfigurasi API")
gemini_api_key = st.sidebar.text_input(
    "Masukkan Gemini API Key", 
    type="password", 
    help="Dapatkan API key gratis di Google AI Studio untuk mengaktifkan fitur Feedback naratif AI."
)

# --- MENGUNDUH & ME-LOAD MODEL EMBEDDING (SENTENCE-BERT) ---
@st.cache_resource
def load_embedding_model():
    # Menggunakan model SBERT Multilingual yang ringan (~120MB) namun sangat akurat untuk semantic search
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model

with st.spinner("Memuat Model NLP (Sentence-BERT)... Harap tunggu sebentar."):
    model_sbert = load_embedding_model()

# --- SIMULASI DATABASE JUDUL TESIS TERDAHULU (STATE-OF-THE-ART) ---
# Pada produksi asli, Anda bisa mengganti list ini dengan hasil pembacaan dari database SQL atau file CSV historis internal prodi.
DATABASE_HISTORIS = [
    "Analisis Sentimen Kebijakan Kampus Merdeka Menggunakan Naive Bayes dan KNN",
    "Penerapan Algoritma Genetika untuk Optimasi Jadwal Kuliah di Universitas Pamulang",
    "Sistem Pakar Diagnosa Penyakit Tanaman Padi Berbasis Web dengan Forward Chaining",
    "Implementasi Kriptografi AES dan RSA untuk Pengamanan Data Rekam Medis Elektronik",
    "Pengembangan Sistem Keamanan Jaringan Menggunakan Intrusion Detection System (IDS)",
    "Optimasi Distribusi Logistik Menggunakan Algoritma Semut (Ant Colony Optimization)",
    "Prediksi Kelulusan Mahasiswa Tepat Waktu Menggunakan Algoritma Random Forest",
    "Sistem Pendukung Keputusan Pemilihan Dosen Terbaik Berbasis Metode AHP dan TOPSIS"
]

# --- PIPELINE EVALUASI MODEL NYATA ---
def pipeline_evaluasi(judul_baru, deskripsi_baru):
    teks_baru = f"{judul_baru} {deskripsi_baru}".strip()
    
    # 1. Proses Embedding Model 1 (Sentence-BERT)
    embedding_baru = model_sbert.encode([teks_baru])
    embeddings_historis = model_sbert.encode(DATABASE_HISTORIS)
    
    # 2. Hitung Jarak Semantik menggunakan Cosine Similarity
    similarities = cosine_similarity(embedding_baru, embeddings_historis)[0]
    idx_paling_mirip = np.argmax(similarities)
    skor_kemiripan_tertinggi = similarities[idx_paling_mirip]
    judul_terdekat = DATABASE_HISTORIS[idx_paling_mirip]
    
    # 3. Pemetaan ke Faktor Penilaian Tesis S2 (Skala 0-100)
    # Skor Novelty berbanding terbalik dengan nilai kemiripan semantik
    novelty_score = int((1 - skor_kemiripan_tertinggi) * 100)
    novelty_score = max(15, min(98, novelty_score)) # Batasi batas bawah dan atas pembobotan
    
    # Heuristik penilaian faktor pendukung lainnya berbasis karakteristik teks
    relevansi_industri = 90 if any(x in teks_baru.lower() for x in ['optimasi', 'machine learning', 'deep learning', 'iot', 'security', 'alloy']) else 70
    kompleksitas_metode = 85 if len(teks_baru.split()) > 15 else 65
    dampak_praktis = 88 if any(x in teks_baru.lower() for x in ['efisiensi', 'performa', 'biaya', 'akurasi']) else 72
    kesesuaian_s2 = 90 if len(judul_baru.split()) >= 8 else 68
    
    skor_faktor = {
        "Novelty (Kebaruan)": novelty_score,
        "Relevansi Industri": relevansi_industri,
        "Kompleksitas Metode": kompleksitas_metode,
        "Dampak Praktis": dampak_praktis,
        "Ksesuaian Standar S2": kesesuaian_s2
    }
    
    # Menentukan Threshold Status Kelayakan (Batas toleransi kemiripan: 0.60 atau 60%)
    if skor_kemiripan_tertinggi > 0.60:
        status = "DITOLAK"
    else:
        status = "BOLEH DIAJUKAN"
        
    return status, skor_faktor, skor_kemiripan_tertinggi, judul_terdekat

# --- PIPELINE GENERATIVE AI (GEMINI GENERATION) ---
def panggil_generative_feedback(judul, status, skor_faktor, judul_mirip, skor_mirip):
    if not gemini_api_key:
        return "⚠️ *Fitur Umpan Balik Otomatis Terbatas*: Masukkan **Gemini API Key** di menu sidebar sebelah kiri untuk memunculkan narasi analisis dosen pakar secara real-time dari LLM."
    
    try:
        genai.configure(api_key=gemini_api_key)
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Bertindaklah sebagai Dosen Komite Evaluator Tesis S2 Magister Teknik Informatika Universitas Pamulang.
        Berikan narasi analisis akademik singkat (maksimal 2 paragraf) mengenai kelayakan judul berikut:
        
        Judul yang Diajukan Mahasiswa: "{judul}"
        Status Keputusan Sistem: {status}
        Detail Matriks Nilai (Skala 100): {skor_faktor}
        Hasil Pengecekan NLP: Judul ini terdeteksi memiliki kemiripan semantik sebesar {round(skor_mirip*100, 2)}% dengan judul tesis terdahulu di database prodi, yaitu: "{judul_mirip}".
        
        Berikan umpan balik yang konstruktif, bernada ilmiah, tegas namun solutif (jika ditolak beri saran perbaikan variabel/metode, jika diterima berikan penguatan). Gunakan Bahasa Indonesia yang formal dan baku.
        """
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gagal terhubung dengan server Gemini API: {str(e)}"

# --- FUNGSI VISUALISASI RADAR CHART ---
def tampilkan_grafik_radar(skor_faktor):
    df_viz = pd.DataFrame(dict(
        Skor=list(skor_faktor.values()),
        Faktor=list(skor_faktor.keys())
    ))
    fig = px.line_polar(df_viz, r='Skor', theta='Faktor', line_close=True, 
                        range_r=[0,100], markers=True,
                        color_discrete_sequence=['#005088'])
    fig.update_traces(fill='toself', fillcolor='rgba(0, 80, 136, 0.25)')
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=50, r=50, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)


# --- ANTARMUKA PENGGUNA (TABS UI) ---
tab1, tab2, tab3 = st.tabs(["✍️ Input Manual", "📄 Upload File DOCX", "📊 Upload File CSV (Massal)"])

# TAB 1: INPUT MANUAL
with tab1:
    st.markdown("### Evaluasi Judul Tunggal")
    with st.form("manual_form"):
        input_judul = st.text_input("Usulan Judul Tesis", 
                                    value="Optimasi Alokasi Kontainer Menggunakan Algoritma Genetika Guna Efisiensi Biaya Operasional")
        input_abstrak = st.text_area("Deskripsi Singkat / Abstrak (Opsional)")
        submit_manual = st.form_submit_button("Jalankan Pipeline Evaluasi AI")
        
    if submit_manual and input_judul:
        with st.spinner("Pipeline berjalan: Menghitung vektor semantik & memanggil GenAI..."):
            status, skor_faktor, nilai_sim, judul_mirip = pipeline_evaluasi(input_judul, input_abstrak)
            feedback_narasi = panggil_generative_feedback(input_judul, status, skor_faktor, judul_mirip, nilai_sim)
            
            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.markdown("#### Hasil Keputusan Sistem:")
                if status == "BOLEH DIAJUKAN":
                    st.success(f"**STATUS AKHIR: {status}** ✅")
                else:
                    st.error(f"**STATUS AKHIR: {status}** ❌")
                
                st.warning(f"**Kemiripan Tertinggi Terdeteksi:** {round(nilai_sim * 100, 2)}% terhadap judul: *\"{judul_mirip}\"*")
                st.markdown("#### Detail Komparasi Metrik:")
                for faktor, skor in skor_faktor.items():
                    st.write(f"{faktor}: **{skor}/100**")
                    st.progress(skor / 100.0)
            
            with col2:
                st.markdown("#### Visualisasi Analisis Faktor (Radar Chart)")
                tampilkan_grafik_radar(skor_faktor)
                
            st.markdown("---")
            st.markdown("#### 💬 Analisis Kontekstual & Rekomendasi (Generative AI Dosen Pakar):")
            st.info(feedback_narasi)

# TAB 2: UPLOAD DOCX
with tab2:
    st.markdown("### Ekstraksi dan Evaluasi dari Dokumen Word")
    file_docx = st.file_uploader("Pilih file DOCX", type=["docx"])
    
    if file_docx is not None:
        if st.button("Proses Dokumen"):
            doc = docx.Document(file_docx)
            full_text = [para.text for para in doc.paragraphs if para.text.strip() != ""]
            
            if len(full_text) > 0:
                ext_judul = full_text[0]
                ext_deskripsi = " ".join(full_text[1:]) if len(full_text) > 1 else ""
                
                st.text_input("Judul Terdeteksi dari File:", value=ext_judul, disabled=True)
                
                status, skor_faktor, nilai_sim, judul_mirip = pipeline_evaluasi(ext_judul, ext_deskripsi)
                feedback_narasi = panggil_generative_feedback(ext_judul, status, skor_faktor, judul_mirip, nilai_sim)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if status == "BOLEH DIAJUKAN":
                        st.success(f"**Status Kelayakan: {status}**")
                    else:
                        st.error(f"**Status Kelayakan: {status}**")
                    st.write(f"**Indeks Kemiripan NLP:** {round(nilai_sim*100, 2)}% dengan tesis terdahulu.")
                    st.markdown("##### Feedback Akademik:")
                    st.write(feedback_narasi)
                with col_b:
                    tampilkan_grafik_radar(skor_faktor)
            else:
                st.warning("File Word tidak memiliki teks yang valid.")

# TAB 3: UPLOAD CSV (MASSAL)
with tab3:
    st.markdown("### Evaluasi Banyak Judul Sekaligus")
    file_csv = st.file_uploader("Pilih file CSV", type=["csv"])
    
    if file_csv is not None:
        if st.button("Proses CSV Massal"):
            df = pd.read_csv(file_csv)
            col_judul = [c for c in df.columns if c.lower() == 'judul']
            
            if col_judul:
                kolom_target = col_judul[0]
                hasil_status, skor_novelty, judul_terdekat, nilai_persen = [], [], [], []
                
                for index, row in df.iterrows():
                    jdl = str(row[kolom_target])
                    stat, skor_faktor, nilai_sim, jdl_mirip = pipeline_evaluasi(jdl, "")
                    
                    hasil_status.append(stat)
                    skor_novelty.append(skor_faktor["Novelty (Kebaruan)"])
                    judul_terdekat.append(jdl_mirip)
                    nilai_persen.append(f"{round(nilai_sim*100, 1)}%")
                    
                df['Rekomendasi_Sistem'] = hasil_status
                df['Skor_Novelty'] = skor_novelty
                df['Judul_Pembanding_Terdekat'] = judul_terdekat
                df['Persentase_Kemiripan_Semantik'] = nilai_persen
                
                st.success("Evaluasi massal berbasis model S-BERT selesai dilakukan!")
                st.dataframe(df, use_container_width=True)
                
                csv_hasil = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Unduh Hasil Evaluasi Massal (CSV)",
                    data=csv_hasil,
                    file_name='laporan_evaluasi_masal_ai.csv',
                    mime='text/csv',
                )
            else:
                st.error("File CSV harus memiliki kolom bernama 'Judul' agar dapat diekstrak oleh model NLP.")
