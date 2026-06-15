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

# HARAP PASTIKAN URL & KEY INI BENAR
SUPABASE_URL = "https://dzatrsuzjyehrsynjvvm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR6YXRyc3V6anllaHJzeW5qdnZtIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3ODQ4NjM4MiwiZXhwIjoyMDk0MDYyMzgyfQ.yyaHi8zOeobXgucE1B2JVBM-oxD49s5vFnG5nmdceGs"

@app.get("/")
def read_root():
    return {"status": "Online", "message": "API Komputasi K-Means Aktif!"}

@app.get("/proses-klaster")
def proses_data(k: int = 3):
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        
        def tarik_data(nama_tabel):
            req = urllib.request.Request(f"{SUPABASE_URL}/rest/v1/{nama_tabel}?select=*", headers=headers)
            try:
                with urllib.request.urlopen(req) as response:
                    return json.loads(response.read().decode()), 200
            except urllib.error.HTTPError as e:
                return e.read().decode(), e.code
            except Exception as e:
                return str(e), 500

        data, status_code = tarik_data("kuesioner")
        if status_code != 200:
            data, status_code = tarik_data("kuisioner")
            
        if status_code != 200 or not data:
            return {"status": "error", "detail": "Data kosong atau gagal ditarik dari database."}
            
        df = pd.DataFrame(data)
        df.columns = df.columns.str.upper()
        kolom_p = [f"P{i}" for i in range(1, 21)]
        
        kolom_ada = [col for col in kolom_p if col in df.columns]
        for col in kolom_ada:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df_bersih = df.dropna(subset=kolom_ada).copy()

        if len(df_bersih) < k:
            return {"status": "error", "detail": f"Data kurang. Sisa: {len(df_bersih)} baris. Butuh {k} baris."}

        df_numeric = df_bersih[kolom_ada]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)
        
        # Ekstrak 3 Dimensi untuk Grafik 3D
        pca = PCA(n_components=3)
        pca_data = pca.fit_transform(scaled_data)
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(pca_data)

        df_bersih['PC1'] = pca_data[:, 0].tolist()
        df_bersih['PC2'] = pca_data[:, 1].tolist()
        df_bersih['PC3'] = pca_data[:, 2].tolist()
        df_bersih['CLUSTER'] = clusters.tolist()
        
        kolom_wajib = ['NAMA', 'KELAS', 'JURUSAN', 'JENIS_KELAMIN']
        for col in kolom_wajib:
            if col not in df_bersih.columns:
                df_bersih[col] = "Tidak Diketahui"

        # Kemas seluruh data agar Javascript bisa nge-filter dan ngebedah P1-P20
        data_lengkap = df_bersih.to_dict(orient='records')
        
        return {
            "status": "success",
            "jumlah_responden_diproses": len(df_numeric),
            "dimensi_setelah_pca": int(pca.n_components_),
            "variansi_kumulatif": round(sum(pca.explained_variance_ratio_) * 100, 2),
            "data_lengkap": data_lengkap
        }

    except Exception as e:
        return {"status": "error", "detail": f"Error Internal Python: {str(e)}"}
