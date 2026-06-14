from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

app = FastAPI()

# Buka jalur CORS biar web frontend Cloudflare lu diizinkan nembak API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Supabase (Pakai Environment Variables biar aman pas di-hosting)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "URL_SUPABASE_LU")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "KEY_SUPABASE_LU")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except:
    supabase = None

@app.get("/")
def read_root():
    return {"status": "Online", "message": "API Komputasi K-Means SMKN 1 Dawuan Aktif!"}

@app.get("/proses-klaster")
def proses_data(k: int = 3):
    if not supabase:
        raise HTTPException(status_code=500, detail="Koneksi Supabase belum di-set")

    # 1. Tarik Data Mentah dari Supabase
    response = supabase.table("kuesioner").select("*").execute()
    data = response.data
    
    if not data:
        raise HTTPException(status_code=404, detail="Database kuesioner masih kosong")
        
    df = pd.DataFrame(data)
    
    # 2. Pre-processing: Ambil kolom P1 sampai P20 dan hapus data kosong
    kolom_p = [f"P{i}" for i in range(1, 21) if f"P{i}" in df.columns]
    
    if not kolom_p:
        raise HTTPException(status_code=400, detail="Kolom kuesioner (P1-P20) tidak ditemukan")

    df_numeric = df[kolom_p].dropna()
    
    # Standarisasi Z-Score
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # 3. Reduksi Dimensi (PCA) - Target > 72% sesuai standar draf TA
    pca = PCA(n_components=0.72)
    pca_data = pca.fit_transform(scaled_data)
    
    # 4. K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(pca_data)
    
    # Gabungin hasil klaster ke dataframe asli
    df_numeric['Cluster'] = clusters.tolist()
    
    return {
        "status": "success",
        "jumlah_responden_diproses": len(df_numeric),
        "dimensi_setelah_pca": int(pca.n_components_),
        "variansi_kumulatif": round(sum(pca.explained_variance_ratio_) * 100, 2),
        "hasil_klaster_per_siswa": df_numeric['Cluster'].tolist()
    }
