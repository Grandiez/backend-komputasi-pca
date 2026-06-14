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

@app.get("/")
def read_root():
    return {"status": "Online", "message": "API Komputasi K-Means SMKN 1 Dawuan Aktif!"}

@app.get("/proses-klaster")
def proses_data(k: int = 3):
    try:
        if not supabase:
            return {"status": "error", "detail": "Koneksi Supabase belum di-set"}

        # 1. Tarik Data Mentah
        response = supabase.table("kuisioner").select("*").execute()
        data = response.data
        
        if not data or len(data) < k:
            return {"status": "error", "detail": f"Data kurang! Minimal butuh {k} siswa untuk dibagi {k} klaster."}
            
        df = pd.DataFrame(data)
        
        # 2. Pre-processing
        kolom_p = [f"P{i}" for i in range(1, 21) if f"P{i}" in df.columns]
        df_numeric = df[kolom_p].dropna()
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)
        
        # 3. PCA (Target > 72%)
        pca = PCA(n_components=0.72)
        pca_data = pca.fit_transform(scaled_data)
        
        # 4. K-Means
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(pca_data)

        # 5. Ekstraksi Koordinat buat Grafik Plotly
        pc1 = pca_data[:, 0].tolist()
        pc2 = pca_data[:, 1].tolist() if pca.n_components_ > 1 else [0] * len(pca_data)
        nama_list = df['Nama'].tolist() if 'Nama' in df.columns else [f"Siswa {i+1}" for i in range(len(df))]
        
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
        return {"status": "error", "detail": str(e)}
