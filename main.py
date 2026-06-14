from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "URL_SUPABASE_LU")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "KEY_SUPABASE_LU")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except:
    supabase = None

@app.get("/proses-klaster")
def proses_data(k: int = 3):
    try:
        if not supabase:
            return {"status": "error", "detail": "Koneksi Supabase belum di-set"}

        # 1. Tarik Data Mentah
        response = supabase.table("kuisioner").select("*").execute()
        data = response.data
        
        # Mencegah error jika data kosong
        if not data:
            return {"status": "error", "detail": "Database Supabase benar-benar kosong atau akses diblokir."}
            
        df = pd.DataFrame(data)
        
        # ANTI-BADAI 1: Paksa semua nama kolom jadi HURUF BESAR (P1, P2, dst)
        df.columns = df.columns.str.upper()
        
        # 2. Pre-processing & Pembersihan Data
        kolom_p = [f"P{i}" for i in range(1, 21)]
        
        # Cek apakah kolom P beneran ada
        kolom_hilang = [col for col in kolom_p if col not in df.columns]
        if kolom_hilang:
            return {"status": "error", "detail": f"Gagal menemukan kolom ini di Supabase: {kolom_hilang}"}

        # ANTI-BADAI 2: Paksa isi kolom P1-P20 jadi angka matematika (jika ada huruf, jadikan NaN)
        for col in kolom_p:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # ANTI-BADAI 3: Hapus baris yang punya nilai kosong (NaN), tapi simpan sisa barisnya
        df_bersih = df.dropna(subset=kolom_p).copy()

        # Cek apakah setelah dibersihkan datanya masih cukup buat K-Means
        if len(df_bersih) < k:
            return {"status": "error", "detail": f"Data hancur saat dibersihkan! Dari {len(df)} baris, cuma sisa {len(df_bersih)} yang angkanya valid. Butuh minimal {k} baris."}

        # Mulai Eksekusi Machine Learning
        df_numeric = df_bersih[kolom_p]
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)
        
        # 3. PCA (Target Variansi > 72%)
        pca = PCA(n_components=0.72)
        pca_data = pca.fit_transform(scaled_data)
        
        # 4. K-Means Clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(pca_data)

        # 5. Ekstraksi Koordinat buat Plotly di Frontend
        pc1 = pca_data[:, 0].tolist()
        pc2 = pca_data[:, 1].tolist() if pca.n_components_ > 1 else [0] * len(pca_data)
        
        # Tarik nama siswa, kalau gak ada kasih nama "Siswa 1, 2, 3..."
        if 'NAMA' in df_bersih.columns:
            nama_list = df_bersih['NAMA'].tolist()
        else:
            nama_list = [f"Siswa {i+1}" for i in range(len(df_bersih))]
        
        return {
            "status": "success",
            "jumlah_responden_diproses": len(df_numeric),
            "dimensi_setelah_pca": int(pca.n_components_),
            "variansi_kumulatif": round(sum(pca.explained_variance_ratio_) * 100, 2),
            "plot_data": {
                "x": pc1,
                "y": pc2,
                "cluster": clusters.tolist(),
                "nama": nama_list
            }
        }

    except Exception as e:
        # Menangkap SEMUA jenis error Python biar gak bikin server Crash 500
        return {"status": "error", "detail": f"Error Internal Python: {str(e)}"}
