from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import urllib.request
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = "https://dzatrsuzjyehrsynjvvm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR6YXRyc3V6anllaHJzeW5qdnZtIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3ODQ4NjM4MiwiZXhwIjoyMDk0MDYyMzgyfQ.yyaHi8zOeobXgucE1B2JVBM-oxD49s5vFnG5nmdceGs"

@app.get("/")
def read_root():
    return {"status": "Online", "message": "API Komputasi K-Means Aktif!"}

@app.get("/proses-klaster")
def proses_data(k: int = 3):
    try:
        # 1. JALUR BYPASS MURNI (Anti-Bug Library Supabase)
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        
        # Fungsi internal buat nembak data langsung ke server
        def tarik_data(nama_tabel):
            req = urllib.request.Request(
                f"{SUPABASE_URL}/rest/v1/{nama_tabel}?select=*",
                headers=headers
            )
            try:
                with urllib.request.urlopen(req) as response:
                    return json.loads(response.read().decode()), 200
            except urllib.error.HTTPError as e:
                return e.read().decode(), e.code
            except Exception as e:
                return str(e), 500

        # Mesin otomatis nyoba ejaan pertama
        data, status_code = tarik_data("kuesioner")
        
        # Kalau gagal (tabel gak ada), mesin otomatis nyoba ejaan kedua
        if status_code != 200:
            data, status_code = tarik_data("kuisioner")
            
        # Kalau masih gagal juga, tampilkan error asli dari server
        if status_code != 200:
            return {"status": "error", "detail": f"Gagal narik Supabase. Kode: {status_code}, Pesan: {data}"}
            
        if not data:
            return {"status": "error", "detail": "Data Supabase kosong (0 baris)."}
            
        # 2. PRE-PROCESSING DATA
        df = pd.DataFrame(data)
        df.columns = df.columns.str.upper()
        kolom_p = [f"P{i}" for i in range(1, 21)]
        
        # Cek ketersediaan kolom (P1 sampai P20)
        kolom_ada = [col for col in kolom_p if col in df.columns]
        if len(kolom_ada) < 20:
            return {"status": "error", "detail": f"Kolom tidak lengkap! Hanya menemukan {len(kolom_ada)} kolom pertanyaan."}

        # Paksa jadi angka dan hapus teks nyasar
        for col in kolom_ada:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df_bersih = df.dropna(subset=kolom_ada).copy()

        if len(df_bersih) < k:
            return {"status": "error", "detail": f"Data hancur saat dibersihkan. Sisa: {len(df_bersih)} baris. Butuh {k} baris."}

        # 3. MACHINE LEARNING (PCA & K-MEANS)
        df_numeric = df_bersih[kolom_ada]
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)
        
        pca = PCA(n_components=0.72)
        pca_data = pca.fit_transform(scaled_data)
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(pca_data)

        # 4. EXPORT KOORDINAT UNTUK GRAFIK
        pc1 = pca_data[:, 0].tolist()
        pc2 = pca_data[:, 1].tolist() if pca.n_components_ > 1 else [0] * len(pca_data)
        
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
        return {"status": "error", "detail": f"Error Internal Python: {str(e)}"}
